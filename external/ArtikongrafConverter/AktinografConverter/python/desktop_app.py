#!/usr/bin/env python3
from __future__ import annotations

import csv
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageTk

from chart_models import get_models
from web_backend import (
    PROJECT_ROOT,
    apply_anchor_edits,
    delete_dataset,
    ensure_editor_assets,
    get_dataset,
    get_series_rows,
    import_uploaded_files,
    init_db,
    list_datasets,
    process_many,
    save_manual_edit,
)


APP_BG = "#f4fbff"
CARD_BG = "#ffffff"
TEXT = "#12334f"
ACCENT = "#0d6efd"
GREEN = "#1f9d55"
MUTED = "#5f748a"

MODELS = get_models(PROJECT_ROOT)
RADIATION_MODEL = MODELS["radiation"]


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _clone_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return [dict(r) for r in rows]


def _hour_precise_from_index(i: int) -> str:
    h_cont = 20.0 + (i / 60.0)
    h_wrap = ((h_cont - 1.0) % 24.0) + 1.0
    hh = int(np.floor(h_wrap + 1e-9))
    mm = int(round((h_wrap - hh) * 60.0))
    if mm == 60:
        mm = 0
        hh += 1
    while hh > 24:
        hh -= 24
    while hh <= 0:
        hh += 24
    return f"{hh}.{mm:02d}"


def _dataset_label(row: Dict[str, object]) -> str:
    rid = int(row.get("id", 0))
    src = str(row.get("source_name", "unnamed"))
    status = str(row.get("status", "unknown"))
    mid = "manual" if int(row.get("has_manual_edit", 0) or 0) else "auto"
    return f"[{rid}] {src} | {status} | {mid}"


class LocalUploadFile:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = path.name
        self._data = path.read_bytes()

    def getbuffer(self) -> memoryview:
        return memoryview(self._data)


@dataclass
class EditPayload:
    dataset_id: int
    aligned_path: Path
    rows: List[Dict[str, object]]


class LoadingPopup:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.win = tk.Toplevel(master)
        self.win.withdraw()
        self.win.overrideredirect(True)
        self.win.configure(bg="#1f2d3a")
        self.win.attributes("-topmost", True)

        self.label = tk.Label(
            self.win,
            text="Yukleniyor...",
            fg="white",
            bg="#1f2d3a",
            font=("Helvetica", 12, "bold"),
            padx=20,
            pady=14,
        )
        self.label.pack()

    def show(self, text: str) -> None:
        self.label.configure(text=text)
        self.master.update_idletasks()
        x = self.master.winfo_rootx() + self.master.winfo_width() // 2 - 90
        y = self.master.winfo_rooty() + self.master.winfo_height() // 2 - 24
        self.win.geometry(f"+{x}+{y}")
        self.win.deiconify()
        self.win.lift()

    def hide(self) -> None:
        self.win.withdraw()


class DesktopApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        init_db()
        self.title("Radiation Data Studio - Desktop")
        self.geometry("1600x960")
        self.minsize(1300, 820)
        self.configure(bg=APP_BG)

        self.busy = False
        self.loading = LoadingPopup(self)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook", background=APP_BG, borderwidth=0)
        style.configure("TNotebook.Tab", font=("Helvetica", 10, "bold"), padding=(14, 8))
        style.configure("Treeview", rowheight=24, font=("Helvetica", 10))
        style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"))

        top = tk.Frame(self, bg=APP_BG)
        top.pack(fill="x", padx=12, pady=(10, 6))
        tk.Label(
            top,
            text="Radiation Data Studio",
            font=("Helvetica", 18, "bold"),
            fg=TEXT,
            bg=APP_BG,
        ).pack(anchor="w")
        tk.Label(
            top,
            text="Desktop workflow: Import + Process + Manual Edit + Review",
            font=("Helvetica", 10),
            fg=MUTED,
            bg=APP_BG,
        ).pack(anchor="w", pady=(2, 0))

        self.status_var = tk.StringVar(value="Hazir")
        status_bar = tk.Label(
            self,
            textvariable=self.status_var,
            anchor="w",
            bg="#e9f3fb",
            fg=TEXT,
            font=("Helvetica", 10),
            padx=10,
            pady=5,
        )
        status_bar.pack(side="bottom", fill="x")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.import_tab = ImportTab(self.notebook, self)
        self.edit_tab = EditTab(self.notebook, self)
        self.review_tab = ReviewTab(self.notebook, self)

        self.notebook.add(self.import_tab, text="Import")
        self.notebook.add(self.edit_tab, text="Edit")
        self.notebook.add(self.review_tab, text="Review")

        self.refresh_all_tabs()

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def refresh_all_tabs(self) -> None:
        self.import_tab.refresh()
        self.edit_tab.refresh_dataset_list()
        self.review_tab.refresh_dataset_list()

    def run_task(
        self,
        loading_text: str,
        task: Callable[[], object],
        on_done: Callable[[object], None],
    ) -> None:
        if self.busy:
            return
        self.busy = True
        self.loading.show(loading_text)
        self.set_status(loading_text)

        def worker() -> None:
            result: object = None
            err: Optional[Exception] = None
            try:
                result = task()
            except Exception as e:  # pragma: no cover - UI surface
                err = e

            def finalize() -> None:
                self.busy = False
                self.loading.hide()
                if err is not None:
                    messagebox.showerror("Hata", str(err))
                    self.set_status("Hata olustu")
                    return
                on_done(result)

            self.after(0, finalize)

        threading.Thread(target=worker, daemon=True).start()


class ImportTab(tk.Frame):
    COLUMNS = ("id", "source_name", "status", "processed", "manual", "imported_at")

    def __init__(self, parent: ttk.Notebook, app: DesktopApp) -> None:
        super().__init__(parent, bg=APP_BG)
        self.app = app

        toolbar = tk.Frame(self, bg=APP_BG)
        toolbar.pack(fill="x", padx=10, pady=10)

        self.btn_import = tk.Button(toolbar, text="Dosya Import", command=self.on_import, bg=ACCENT, fg="white", relief="flat")
        self.btn_import.pack(side="left", padx=(0, 8))

        self.btn_process = tk.Button(toolbar, text="Seciliyi Isle", command=self.on_process_selected, relief="flat")
        self.btn_process.pack(side="left", padx=(0, 8))

        self.btn_delete = tk.Button(toolbar, text="Seciliyi Sil", command=self.on_delete_selected, relief="flat")
        self.btn_delete.pack(side="left", padx=(0, 8))

        tk.Label(toolbar, text="Spike Margin", bg=APP_BG, fg=TEXT).pack(side="left", padx=(16, 4))
        self.spike_var = tk.IntVar(value=0)
        self.spike_spin = tk.Spinbox(toolbar, from_=0, to=120, textvariable=self.spike_var, width=5)
        self.spike_spin.pack(side="left")

        tk.Button(toolbar, text="Yenile", command=self.refresh, relief="flat").pack(side="right")

        table_wrap = tk.Frame(self, bg=CARD_BG, bd=1, relief="solid")
        table_wrap.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.tree = ttk.Treeview(table_wrap, columns=self.COLUMNS, show="headings", selectmode="extended")
        for c in self.COLUMNS:
            self.tree.heading(c, text=c)
        self.tree.column("id", width=70, anchor="center")
        self.tree.column("source_name", width=340, anchor="w")
        self.tree.column("status", width=110, anchor="center")
        self.tree.column("processed", width=80, anchor="center")
        self.tree.column("manual", width=80, anchor="center")
        self.tree.column("imported_at", width=180, anchor="center")

        y_scroll = ttk.Scrollbar(table_wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=y_scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        y_scroll.pack(side="right", fill="y")

        self.rows_by_id: Dict[int, Dict[str, object]] = {}

    def refresh(self) -> None:
        rows = list_datasets()
        self.rows_by_id = {int(r["id"]): dict(r) for r in rows}
        self.tree.delete(*self.tree.get_children())
        for r in rows:
            rid = int(r["id"])
            self.tree.insert(
                "",
                "end",
                iid=str(rid),
                values=(
                    rid,
                    str(r.get("source_name", "")),
                    str(r.get("status", "")),
                    int(r.get("processed", 0)),
                    int(r.get("has_manual_edit", 0)),
                    str(r.get("imported_at", "")),
                ),
            )
        self.app.set_status(f"Toplam kayit: {len(rows)}")

    def _selected_ids(self) -> List[int]:
        out: List[int] = []
        for iid in self.tree.selection():
            try:
                out.append(int(iid))
            except Exception:
                continue
        return out

    def on_import(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Resim sec",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return

        files = [LocalUploadFile(Path(p)) for p in paths]

        def task() -> List[int]:
            return import_uploaded_files(files)

        def done(new_ids: object) -> None:
            ids = [int(x) for x in (new_ids or [])]
            self.refresh()
            self.app.refresh_all_tabs()
            self.app.set_status(f"Import tamamlandi: {len(ids)} dosya")
            if ids and messagebox.askyesno("Islem", "Import bitti. Yeni dosyalari simdi islemek ister misin?"):
                self._process_ids(ids)

        self.app.run_task("Yukleniyor... import", task, done)

    def _process_ids(self, ids: Sequence[int]) -> None:
        spike = int(self.spike_var.get())

        def task() -> object:
            return process_many(ids, spike_margin_int=spike)

        def done(result_obj: object) -> None:
            results = list(result_obj or [])
            ok = sum(1 for r in results if getattr(r, "ok", False))
            err = len(results) - ok
            self.refresh()
            self.app.refresh_all_tabs()
            self.app.set_status(f"Isleme tamamlandi: ok={ok}, hata={err}")
            if err:
                errs = [str(getattr(r, "message", "")) for r in results if not getattr(r, "ok", False)]
                messagebox.showwarning("Isleme Sonucu", "\n".join(errs[:8]))

        self.app.run_task("Yukleniyor... secili kayitlar isleniyor", task, done)

    def on_process_selected(self) -> None:
        ids = self._selected_ids()
        if not ids:
            messagebox.showinfo("Bilgi", "Lutfen en az bir kayit sec.")
            return
        self._process_ids(ids)

    def on_delete_selected(self) -> None:
        ids = self._selected_ids()
        if not ids:
            messagebox.showinfo("Bilgi", "Lutfen en az bir kayit sec.")
            return
        if not messagebox.askyesno("Onay", f"{len(ids)} kayit silinecek. Emin misin?"):
            return

        def task() -> int:
            return sum(1 for did in ids if delete_dataset(int(did)))

        def done(deleted_obj: object) -> None:
            deleted = int(deleted_obj or 0)
            self.refresh()
            self.app.refresh_all_tabs()
            self.app.set_status(f"Silindi: {deleted} kayit")

        self.app.run_task("Yukleniyor... secili kayitlar siliniyor", task, done)


class EditTab(tk.Frame):
    def __init__(self, parent: ttk.Notebook, app: DesktopApp) -> None:
        super().__init__(parent, bg=APP_BG)
        self.app = app

        self.dataset_map: Dict[str, int] = {}
        self.dataset_var = tk.StringVar()
        self.stride_var = tk.IntVar(value=5)

        self.dataset_id: Optional[int] = None
        self.base_rows: List[Dict[str, object]] = []
        self.preview_rows: List[Dict[str, object]] = []
        self.control_indices: List[int] = []

        self.scale = 1.0
        self.img_src: Optional[Image.Image] = None
        self.img_tk: Optional[ImageTk.PhotoImage] = None
        self.handle_positions: List[Dict[str, float]] = []
        self.handle_item_to_index: Dict[int, int] = {}
        self.curve_item: Optional[int] = None
        self.active_handle_item: Optional[int] = None
        self.live_job: Optional[str] = None
        self.edit_payload_cache: Dict[int, EditPayload] = {}
        self.undo_stack: List[List[Dict[str, float]]] = []
        self.drag_start_snapshot: Optional[List[Dict[str, float]]] = None

        top = tk.Frame(self, bg=APP_BG)
        top.pack(fill="x", padx=10, pady=10)

        tk.Label(top, text="Kayit", bg=APP_BG, fg=TEXT).pack(side="left")
        self.dataset_combo = ttk.Combobox(top, textvariable=self.dataset_var, state="readonly", width=52)
        self.dataset_combo.pack(side="left", padx=6)

        tk.Button(top, text="Sec", command=self.on_select_dataset, relief="flat").pack(side="left", padx=(0, 10))

        tk.Label(top, text="Yogunluk (dk)", bg=APP_BG, fg=TEXT).pack(side="left", padx=(6, 4))
        self.stride_spin = tk.Spinbox(top, from_=1, to=15, textvariable=self.stride_var, width=4, command=self.on_stride_changed)
        self.stride_spin.pack(side="left")

        self.btn_save = tk.Button(top, text="Kaydet ve Isle", command=self.on_save, relief="flat")
        self._set_save_button_colors()
        self.btn_save.pack(side="left", padx=(10, 0))

        self.hint_var = tk.StringVar(value="Secili kayit yok.")
        tk.Label(self, textvariable=self.hint_var, bg=APP_BG, fg=MUTED, anchor="w").pack(fill="x", padx=10, pady=(0, 6))

        canvas_wrap = tk.Frame(self, bg=CARD_BG, bd=1, relief="solid")
        canvas_wrap.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.canvas = tk.Canvas(canvas_wrap, bg="white", highlightthickness=0, cursor="cross")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll_y = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self.canvas.yview)
        self.scroll_y.pack(side="right", fill="y")
        self.scroll_x = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.scroll_x.pack(fill="x", padx=10, pady=(0, 10))
        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)

        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<KeyPress-z>", self.on_key_undo)
        self.canvas.bind("<KeyPress-Z>", self.on_key_undo)

    def refresh_dataset_list(self) -> None:
        rows = [r for r in list_datasets() if int(r.get("processed", 0)) == 1]
        labels = [_dataset_label(r) for r in rows]
        self.dataset_map = {label: int(r["id"]) for label, r in zip(labels, rows)}
        self.dataset_combo["values"] = labels
        if labels and self.dataset_var.get() not in self.dataset_map:
            self.dataset_var.set(labels[0])

    def _load_payload(self, dataset_id: int) -> EditPayload:
        cached = self.edit_payload_cache.get(dataset_id)
        if cached is not None and cached.rows:
            row = get_dataset(dataset_id)
            cached_csv = str(cached.rows[0].get("_src_csv", ""))
            if row is not None and str(row.get("current_csv_path", "")) == cached_csv:
                return cached

        aligned_path = ensure_editor_assets(dataset_id)
        rows = get_series_rows(dataset_id, use_auto=False)
        if aligned_path is None or not rows:
            raise RuntimeError("Edit verisi bulunamadi.")

        row = get_dataset(dataset_id) or {}
        current_csv = str(row.get("current_csv_path", ""))
        normalized = _clone_rows(rows)
        for rr in normalized:
            rr["_src_csv"] = current_csv

        payload = EditPayload(dataset_id=dataset_id, aligned_path=Path(aligned_path), rows=normalized)
        self.edit_payload_cache[dataset_id] = payload
        return payload

    def on_select_dataset(self) -> None:
        label = self.dataset_var.get().strip()
        if not label or label not in self.dataset_map:
            return
        did = int(self.dataset_map[label])

        def task() -> EditPayload:
            return self._load_payload(did)

        def done(payload_obj: object) -> None:
            payload = payload_obj  # type: ignore[assignment]
            assert isinstance(payload, EditPayload)
            self.dataset_id = payload.dataset_id
            self.base_rows = _clone_rows(payload.rows)
            self.preview_rows = _clone_rows(payload.rows)
            self.undo_stack = []
            self.drag_start_snapshot = None
            self._load_image(payload.aligned_path)
            self._rebuild_control_points_from_preview()
            self._draw_full_scene()
            self.app.set_status(f"Edit kaydi acildi: {self.dataset_id}")
            self.hint_var.set("Noktalari surukleyerek duzelt. Geri almak icin Z tusunu kullan.")

        self.app.run_task("Yukleniyor... edit kaydi aciliyor", task, done)

    def _load_image(self, path: Path) -> None:
        img = Image.open(path).convert("RGB")
        max_w = 1300
        self.scale = min(1.0, max_w / max(float(img.width), 1.0))
        if self.scale < 1.0:
            img = img.resize((int(round(img.width * self.scale)), int(round(img.height * self.scale))), Image.Resampling.LANCZOS)
        self.img_src = img
        self.img_tk = ImageTk.PhotoImage(img)

    def on_stride_changed(self) -> None:
        if self.dataset_id is None or not self.preview_rows:
            return
        self._rebuild_control_points_from_preview()
        self._draw_full_scene()

    def _rebuild_control_points_from_preview(self) -> None:
        stride = max(1, int(self.stride_var.get()))
        n = len(self.preview_rows)
        idx = list(range(0, n, stride))
        if idx and idx[-1] != n - 1:
            idx.append(n - 1)
        self.control_indices = idx

        points: List[Dict[str, float]] = []
        for i in self.control_indices:
            rr = self.preview_rows[int(i)]
            points.append(
                {
                    "x": _safe_float(rr.get("x")) * self.scale,
                    "y": _safe_float(rr.get("y")) * self.scale,
                }
            )
        self.handle_positions = points

    def _draw_full_scene(self) -> None:
        self.canvas.delete("all")
        self.handle_item_to_index.clear()
        self.curve_item = None
        if self.img_tk is None:
            return

        self.canvas.create_image(0, 0, image=self.img_tk, anchor="nw", tags=("bg",))
        self._draw_curve()
        self._draw_handles()
        self.canvas.tag_raise("handle")

        w = self.img_tk.width()
        h = self.img_tk.height()
        self.canvas.configure(scrollregion=(0, 0, w + 10, h + 10))

    def _draw_curve(self) -> None:
        if not self.preview_rows:
            return
        pts: List[float] = []
        for rr in self.preview_rows:
            pts.append(_safe_float(rr.get("x")) * self.scale)
            pts.append(_safe_float(rr.get("y")) * self.scale)
        if len(pts) >= 4:
            self.curve_item = self.canvas.create_line(*pts, fill=GREEN, width=2.2, smooth=True, tags=("curve",))

    def _draw_handles(self) -> None:
        r = 4.5
        for i, hp in enumerate(self.handle_positions):
            x, y = hp["x"], hp["y"]
            item = self.canvas.create_oval(
                x - r,
                y - r,
                x + r,
                y + r,
                fill="#23b26f",
                outline="white",
                width=1.0,
                tags=("handle", f"handle_{i}"),
            )
            self.handle_item_to_index[item] = i

    def _anchors_from_handles(self) -> List[Tuple[int, float]]:
        if not self.base_rows:
            return []
        base_x = np.asarray([_safe_float(r.get("x")) for r in self.base_rows], dtype=np.float64)
        out: Dict[int, float] = {}
        for hp in self.handle_positions:
            x_img = hp["x"] / max(self.scale, 1e-6)
            y_img = hp["y"] / max(self.scale, 1e-6)
            idx = int(np.argmin(np.abs(base_x - x_img)))
            rad = float(RADIATION_MODEL.value_from_xy(np.asarray([base_x[idx]]), np.asarray([y_img]))[0])
            out[idx] = float(np.clip(rad, 0.0, 2.0))
        return sorted(out.items(), key=lambda t: int(t[0]))

    def _recompute_preview(self) -> None:
        if self.dataset_id is None or not self.base_rows:
            return
        anchors = self._anchors_from_handles()
        self.preview_rows = apply_anchor_edits(self.dataset_id, self.base_rows, anchors)
        self.canvas.delete("curve")
        self._draw_curve()
        self.canvas.tag_raise("handle")
        self._set_save_button_colors()

    def _set_save_button_colors(self) -> None:
        self.btn_save.configure(
            bg=ACCENT,
            fg="white",
            activebackground="#0b5ed7",
            activeforeground="white",
            disabledforeground="#d7e7ff",
            relief="flat",
            bd=0,
            highlightthickness=0,
        )

    def _schedule_preview_refresh(self) -> None:
        if self.live_job is not None:
            self.after_cancel(self.live_job)
        self.live_job = self.after(12, self._preview_refresh_job)

    def _preview_refresh_job(self) -> None:
        self.live_job = None
        self._recompute_preview()

    def on_canvas_press(self, event: tk.Event) -> None:
        self.canvas.focus_set()
        item = self.canvas.find_withtag("current")
        if not item:
            self.active_handle_item = None
            return
        iid = int(item[0])
        if iid not in self.handle_item_to_index:
            self.active_handle_item = None
            return
        self.active_handle_item = iid
        self.drag_start_snapshot = [dict(p) for p in self.handle_positions]

    def on_canvas_drag(self, event: tk.Event) -> None:
        if self.active_handle_item is None:
            return
        idx = self.handle_item_to_index.get(self.active_handle_item)
        if idx is None:
            return

        max_w = float((self.img_tk.width() - 1) if self.img_tk is not None else (self.canvas.winfo_width() - 1))
        max_h = float((self.img_tk.height() - 1) if self.img_tk is not None else (self.canvas.winfo_height() - 1))
        x = max(0.0, min(float(event.x), max_w))
        y = max(0.0, min(float(event.y), max_h))
        self.handle_positions[idx]["x"] = x
        self.handle_positions[idx]["y"] = y

        r = 4.5
        self.canvas.coords(self.active_handle_item, x - r, y - r, x + r, y + r)
        self._schedule_preview_refresh()

    def on_canvas_release(self, _event: tk.Event) -> None:
        if self.active_handle_item is None:
            return
        self.active_handle_item = None
        if self.drag_start_snapshot is not None:
            self.undo_stack.append(self.drag_start_snapshot)
            if len(self.undo_stack) > 200:
                self.undo_stack = self.undo_stack[-200:]
        self.drag_start_snapshot = None
        self._recompute_preview()

    def on_key_undo(self, _event: tk.Event) -> None:
        if not self.undo_stack:
            return
        previous = self.undo_stack.pop()
        self.handle_positions = [dict(p) for p in previous]
        self._recompute_preview()
        self.hint_var.set("Son tasima geri alindi (Z).")

    def on_save(self) -> None:
        if self.dataset_id is None or not self.preview_rows:
            return
        did = int(self.dataset_id)
        rows_to_save = _clone_rows(self.preview_rows)

        def task() -> object:
            return save_manual_edit(did, rows_to_save)

        def done(res_obj: object) -> None:
            ok = bool(getattr(res_obj, "ok", False))
            msg = str(getattr(res_obj, "message", ""))
            if ok:
                messagebox.showinfo("Kayit", msg or "Kayit basarili.")
                self.base_rows = _clone_rows(self.preview_rows)
                self.undo_stack = []
                self.drag_start_snapshot = None
                self._rebuild_control_points_from_preview()
                self._draw_full_scene()
                self.app.refresh_all_tabs()
                self.app.set_status(f"Kaydedildi: {did}")
            else:
                messagebox.showerror("Kayit Hatasi", msg or "Kayit basarisiz.")

        self.app.run_task("Yukleniyor... manuel duzenleme kaydediliyor", task, done)


class ReviewTab(tk.Frame):
    def __init__(self, parent: ttk.Notebook, app: DesktopApp) -> None:
        super().__init__(parent, bg=APP_BG)
        self.app = app
        self.dataset_map: Dict[str, int] = {}
        self.dataset_var = tk.StringVar()
        self.diff_img_tk: Optional[ImageTk.PhotoImage] = None
        self.hover_var = tk.StringVar(value="")
        self.cur_hover_points: List[Tuple[float, float, str, float]] = []
        self.current_rows: List[Dict[str, object]] = []
        self.current_source_name: str = ""

        top = tk.Frame(self, bg=APP_BG)
        top.pack(fill="x", padx=10, pady=10)

        tk.Label(top, text="Kayit", bg=APP_BG, fg=TEXT).pack(side="left")
        self.combo = ttk.Combobox(top, textvariable=self.dataset_var, state="readonly", width=52)
        self.combo.pack(side="left", padx=6)
        tk.Button(top, text="Sec", command=self.on_load, relief="flat").pack(side="left", padx=(0, 8))
        tk.Button(top, text="Yenile", command=self.refresh_dataset_list, relief="flat").pack(side="left")
        tk.Button(top, text="Tabloyu Indir (CSV)", command=self.on_export_csv, bg=ACCENT, fg="white", relief="flat").pack(side="right")

        body = tk.PanedWindow(self, orient="horizontal", sashrelief="raised", bg=APP_BG)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left = tk.Frame(body, bg=CARD_BG, bd=1, relief="solid")
        right = tk.Frame(body, bg=CARD_BG, bd=1, relief="solid")
        body.add(left, minsize=800)
        body.add(right, minsize=500)

        self.plot = tk.Canvas(left, bg="white", highlightthickness=0)
        self.plot.pack(fill="both", expand=True)
        self.plot.bind("<Motion>", self.on_plot_hover)
        self.plot.bind("<Leave>", self.on_plot_leave)

        tk.Label(right, text="Diff", bg=CARD_BG, fg=TEXT, font=("Helvetica", 11, "bold")).pack(anchor="w", padx=8, pady=6)
        self.diff_label = tk.Label(right, bg=CARD_BG)
        self.diff_label.pack(fill="x", padx=8)

        tk.Label(left, textvariable=self.hover_var, bg=CARD_BG, fg=TEXT, anchor="w").pack(fill="x", padx=8, pady=(4, 8))

        table_wrap = tk.Frame(right, bg=CARD_BG)
        table_wrap.pack(fill="both", expand=True, padx=8, pady=8)
        self.table = ttk.Treeview(table_wrap, columns=("hour", "rad", "x", "y"), show="headings")
        for c in ("hour", "rad", "x", "y"):
            self.table.heading(c, text=c)
        self.table.column("hour", width=90, anchor="center")
        self.table.column("rad", width=90, anchor="center")
        self.table.column("x", width=90, anchor="center")
        self.table.column("y", width=90, anchor="center")
        y_scroll = ttk.Scrollbar(table_wrap, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=y_scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        y_scroll.pack(side="right", fill="y")

    def refresh_dataset_list(self) -> None:
        rows = [r for r in list_datasets() if int(r.get("processed", 0)) == 1]
        labels = [_dataset_label(r) for r in rows]
        self.dataset_map = {label: int(r["id"]) for label, r in zip(labels, rows)}
        self.combo["values"] = labels
        if labels and self.dataset_var.get() not in self.dataset_map:
            self.dataset_var.set(labels[0])

    def on_load(self) -> None:
        label = self.dataset_var.get().strip()
        if not label or label not in self.dataset_map:
            return
        did = int(self.dataset_map[label])

        def task() -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
            row = get_dataset(did)
            if row is None:
                raise RuntimeError("Kayit bulunamadi.")
            auto_rows = get_series_rows(did, use_auto=True)
            cur_rows = get_series_rows(did, use_auto=False)
            return row, auto_rows, cur_rows

        def done(payload: object) -> None:
            row, auto_rows, cur_rows = payload  # type: ignore[misc]
            self.current_rows = [dict(r) for r in cur_rows]
            self.current_source_name = str(row.get("source_name", "series"))
            self._draw_plot(auto_rows, cur_rows)
            self._draw_diff(row)
            self._fill_table(cur_rows)
            self.app.set_status(f"Inceleme yuklendi: {did}")

        self.app.run_task("Yukleniyor... inceleme verisi", task, done)

    def _draw_plot(self, auto_rows: Sequence[Dict[str, object]], cur_rows: Sequence[Dict[str, object]]) -> None:
        self.plot.delete("all")
        self.cur_hover_points = []
        self.hover_var.set("")
        w = max(640, self.plot.winfo_width())
        h = max(360, self.plot.winfo_height())
        margin_l, margin_t, margin_r, margin_b = 48, 18, 18, 28
        pw = w - margin_l - margin_r
        ph = h - margin_t - margin_b
        if pw <= 10 or ph <= 10:
            return

        self.plot.create_rectangle(margin_l, margin_t, margin_l + pw, margin_t + ph, outline="#d3e3f2")
        for i in range(0, 25):
            x = margin_l + (pw * i / 24.0)
            self.plot.create_line(x, margin_t, x, margin_t + ph, fill="#eef5fb")
            hour = 20 + i
            hour = ((hour - 1) % 24) + 1
            self.plot.create_text(x, margin_t + ph + 12, text=str(hour), fill=MUTED, font=("Helvetica", 8))
        for j in range(0, 5):
            y = margin_t + (ph * j / 4.0)
            val = 2.0 - (2.0 * j / 4.0)
            self.plot.create_line(margin_l, y, margin_l + pw, y, fill="#eef5fb")
            self.plot.create_text(margin_l - 18, y, text=f"{val:.1f}", fill=MUTED, font=("Helvetica", 8))

        def row_to_xy(rows: Sequence[Dict[str, object]], collect_hover: bool = False) -> List[float]:
            out: List[float] = []
            n = max(len(rows), 1)
            for i, r in enumerate(rows):
                x = margin_l + (pw * i / max(1, n - 1))
                rad = _safe_float(r.get("radiation"))
                rad = max(0.0, min(2.0, rad))
                y = margin_t + ((2.0 - rad) / 2.0) * ph
                out.extend([x, y])
                if collect_hover:
                    hour_precise = str(r.get("hour_precise", _hour_precise_from_index(i)))
                    self.cur_hover_points.append((x, y, hour_precise, rad))
            return out

        a = row_to_xy(auto_rows, collect_hover=False)
        c = row_to_xy(cur_rows, collect_hover=True)
        if len(a) >= 4:
            self.plot.create_line(*a, fill="#8bb6de", width=1.6, smooth=True)
        if len(c) >= 4:
            self.plot.create_line(*c, fill="#0d6efd", width=2.2, smooth=True)

    def on_plot_hover(self, event: tk.Event) -> None:
        if not self.cur_hover_points:
            return
        x = float(event.x)
        y = float(event.y)
        arr = np.asarray([(p[0], p[1]) for p in self.cur_hover_points], dtype=np.float64)
        d2 = np.square(arr[:, 0] - x) + np.square(arr[:, 1] - y)
        idx = int(np.argmin(d2))
        px, py, hour_txt, rad = self.cur_hover_points[idx]

        if abs(px - x) > 40 and abs(py - y) > 40:
            self.hover_var.set("")
            self.plot.delete("hover")
            return

        self.hover_var.set(f"Saat: {hour_txt}   Radiation: {rad:.4f}")
        self.plot.delete("hover")
        self.plot.create_oval(px - 4, py - 4, px + 4, py + 4, fill="#1f9d55", outline="white", width=1, tags=("hover",))

    def on_plot_leave(self, _event: tk.Event) -> None:
        self.hover_var.set("")
        self.plot.delete("hover")

    def _draw_diff(self, row: Dict[str, object]) -> None:
        p = Path(str(row.get("current_diff_path", "")))
        if not p.exists():
            self.diff_label.configure(image="", text="Diff bulunamadi", fg=MUTED)
            self.diff_img_tk = None
            return
        img = Image.open(p).convert("RGB")
        img = ImageOps.contain(img, (560, 320), Image.Resampling.LANCZOS)
        self.diff_img_tk = ImageTk.PhotoImage(img)
        self.diff_label.configure(image=self.diff_img_tk, text="")

    def _fill_table(self, rows: Sequence[Dict[str, object]]) -> None:
        self.table.delete(*self.table.get_children())
        for r in rows:
            self.table.insert(
                "",
                "end",
                values=(
                    str(r.get("hour_precise", "")),
                    f"{_safe_float(r.get('radiation')):.4f}",
                    f"{_safe_float(r.get('x')):.1f}",
                    f"{_safe_float(r.get('y')):.1f}",
                ),
            )

    def on_export_csv(self) -> None:
        if not self.current_rows:
            messagebox.showinfo("Bilgi", "Indirmek icin once bir kayit secin.")
            return

        base_name = Path(self.current_source_name or "series").stem
        default_name = f"{base_name}_review.csv"
        out_path = filedialog.asksaveasfilename(
            title="Tabloyu CSV olarak kaydet",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not out_path:
            return

        headers = ("source_file", "hour_precise", "radiation", "x", "y")
        with Path(out_path).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(headers))
            w.writeheader()
            for r in self.current_rows:
                w.writerow(
                    {
                        "source_file": str(r.get("source_file", self.current_source_name)),
                        "hour_precise": str(r.get("hour_precise", "")),
                        "radiation": f"{_safe_float(r.get('radiation')):.4f}",
                        "x": f"{_safe_float(r.get('x')):.3f}",
                        "y": f"{_safe_float(r.get('y')):.3f}",
                    }
                )
        self.app.set_status(f"CSV indirildi: {out_path}")


def main() -> None:
    app = DesktopApp()
    app.mainloop()


if __name__ == "__main__":
    main()
