const { chromium } = require('playwright');
const crypto = require('crypto');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('http://127.0.0.1:4173/index.html', { waitUntil: 'domcontentloaded' });

  const getHash = async () => page.$$eval('#sim-svg path', els => {
    const str = els.map(e => e.getAttribute('d')).filter(Boolean).join('|');
    return str;
  });

  const before = await getHash();

  await page.$eval('#sim-rain', el => { el.value = 15; el.dispatchEvent(new Event('input', { bubbles: true })); });
  await page.$eval('#sim-et0', el => { el.value = -10; el.dispatchEvent(new Event('input', { bubbles: true })); });
  await page.waitForTimeout(200);
  const after = await getHash();

  const hash = s => crypto.createHash('md5').update(s).digest('hex');
  console.log('hash before', hash(before), 'hash after', hash(after), 'changed', hash(before) !== hash(after));

  await browser.close();
})();
