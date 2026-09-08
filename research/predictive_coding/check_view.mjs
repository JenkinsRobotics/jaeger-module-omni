// Optional developer UI check; Playwright is not a demo dependency.
// node check_view.mjs /absolute/path/to/playwright/index.mjs
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {fileURLToPath, pathToFileURL} from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const {chromium} = await import(pathToFileURL(path.resolve(process.argv[2])).href);
const browser = await chromium.launch({headless:true,channel:'chrome'});
const checks = [];
const screenshots = path.join(root,'artifacts/predictive-coding/screenshots');
await mkdir(screenshots,{recursive:true});
try {
  for (const width of [360,736]) for (const colorScheme of ['light','dark']) {
    const page = await browser.newPage({viewport:{width,height:1000},colorScheme});
    const errors = [];
    page.on('pageerror',error=>errors.push(error.message));
    await page.goto(pathToFileURL(path.join(root,'artifacts/predictive-coding/demo.html')).href);
    assert.equal(await page.locator('.sensor').count(),12);
    const initial = await page.locator('[data-output="energy"]').innerText();
    await page.locator('[data-control="sensory-step"]').fill('40');
    const settled = await page.locator('[data-output="energy"]').innerText();
    assert.notEqual(initial,settled);
    await page.locator('[data-control="prior"]').selectOption('strong_wrong');
    assert.notEqual(await page.locator('[data-output="energy"]').innerText(),settled);
    await page.screenshot({path:path.join(screenshots,`sensory-${width}-${colorScheme}.png`),fullPage:true});
    await page.locator('[data-control="demo"]').selectOption('world');
    await page.locator('[data-control="world-step"]').fill('25');
    assert.equal(await page.locator('svg circle').count(),3);
    const before = await page.locator('[data-output="world-metrics"]').innerText();
    await page.locator('[data-control="budget"]').selectOption('backprop_matched_time');
    assert.notEqual(await page.locator('[data-output="world-metrics"]').innerText(),before);
    const standard = await page.locator('[data-output="world-metrics"]').innerText();
    await page.locator('[data-control="world"]').selectOption('shifted_test');
    assert.notEqual(await page.locator('[data-output="world-metrics"]').innerText(),standard);
    await page.locator('[data-control="seed"]').selectOption('2');
    assert.equal(await page.locator('[data-control="seed"]').inputValue(),'2');
    await page.screenshot({path:path.join(screenshots,`world-${width}-${colorScheme}.png`),fullPage:true});
    const overflow = await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth);
    assert.equal(overflow,false,JSON.stringify(await page.evaluate(()=>Array.from(document.querySelectorAll('body *'))
      .filter(n=>n.getBoundingClientRect().right>innerWidth).map(n=>({tag:n.tagName,control:n.dataset.control,width:n.getBoundingClientRect().width})))));
    const clipped = await page.locator('svg text').evaluateAll(nodes => nodes.some(n => {
      const b=n.getBBox(), width=n.ownerSVGElement.viewBox.baseVal.width;
      return b.x<0 || b.x+b.width>width;
    }));
    assert.equal(clipped,false);
    assert.deepEqual(errors,[]);
    checks.push({width,colorScheme,passed:true,page_errors:errors});
    await page.close();
  }
} finally { await browser.close(); }
const html = await readFile(path.join(root,'artifacts/predictive-coding/demo.html'));
const {createHash} = await import('node:crypto');
const evidence = {command:process.argv,checks,html_sha256:createHash('sha256').update(html).digest('hex')};
await writeFile(path.join(root,'research/candidate_evals/predictive-coding-ui-20260907.json'),JSON.stringify(evidence,null,2)+'\n');
console.log(JSON.stringify(evidence,null,2));
