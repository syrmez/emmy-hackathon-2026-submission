// node tests/test_softmax_browser.mjs — Node 22+ and installed Chrome; no packages.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtempSync, existsSync, readFileSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {once} from 'node:events';
const profile=mkdtempSync(join(tmpdir(),'softmax-layout-'));
const browser=spawn(process.env.CHROME||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',[
  '--headless=new','--no-first-run','--no-default-browser-check','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'
],{stdio:'ignore'});
const exited=once(browser,'exit'), sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
let ws;
try {
  for(let i=0;i<100&&!existsSync(join(profile,'DevToolsActivePort'));i++)await sleep(100);
  const port=readFileSync(join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];
  const pages=await(await fetch(`http://127.0.0.1:${port}/json`)).json();
  ws=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);await once(ws,'open');
  let id=0;const pending=new Map();
  ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(pending.has(m.id)){const [resolve,reject]=pending.get(m.id);pending.delete(m.id);m.error?reject(Error(JSON.stringify(m.error))):resolve(m.result);}});
  const call=(method,params={})=>new Promise((resolve,reject)=>{pending.set(++id,[resolve,reject]);ws.send(JSON.stringify({id,method,params}));});
  const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true});assert.ok(!r.exceptionDetails,JSON.stringify(r.exceptionDetails));return r.result.value;};
  await call('Page.navigate',{url:new URL('../softmax.html',import.meta.url).href});
  for(let i=0;i<100;i++){if(await evaluate('typeof replay!=="undefined"&&!!replay'))break;await sleep(100);}
  for(const [width,height] of [[1366,768],[1440,800],[1280,720]]) {
    await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
    for(let step=0;step<26;step++) {
      await evaluate(`go(${step})`);
      assert.ok(await evaluate(`document.documentElement.scrollHeight<=${height}`),`${width}×${height}, step ${step}: vertical overflow ${await evaluate('JSON.stringify({width:innerWidth,sections:[...document.querySelector("main").children].map(e=>[e.tagName,e.getBoundingClientRect().height]),zones:[...document.querySelectorAll(".zone")].map(e=>[...e.children].map(c=>[c.className,c.getBoundingClientRect().height]))})')}`);
      assert.ok(await evaluate(`document.documentElement.scrollWidth<=${width}`),'horizontal overflow');
    }
  }
  await evaluate('$("recording").click()');
  for(const [width,height] of [[1280,640],[1366,768],[1440,900]]) {
    await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
    for(const view of ['replay','compiler','inspect']) {
      await evaluate(`$('recording-view').value='${view}';$('recording-view').dispatchEvent(new Event('change'))`);
      for(let step=0;step<26;step++) {
        await evaluate(`go(${step})`);
        assert.ok(await evaluate(`document.documentElement.scrollHeight<=${height}&&document.documentElement.scrollWidth<=${width}`),`recording ${view} ${width}×${height} step ${step}: page overflow`);
      }
      const target={replay:'replay-panel',compiler:'compiler-panel',inspect:'inspection'}[view];
      assert.ok(await evaluate(`$('${target}').getBoundingClientRect().height>200`),'selected view has usable height');
    }
  }
  await evaluate('$("recording").click()');
  assert.equal(await evaluate('document.querySelector(".replay-details").open'),false,'restore disclosure state');
  await evaluate('document.querySelector(".replay-details").open=true;selectThread(37);go(6)');
  assert.equal(await evaluate('$("thread-id").value'),'37');
  for(const [step,source,ptx] of [[2,'def address','ld.global.f32'],[6,'self.combine(stmt)','shfl.sync.bfly.b32'],[13,'def tree','bar.sync 0']]) {
    await evaluate(`go(${step})`);
    assert.ok(await evaluate(`$('compiler-source').textContent.includes(${JSON.stringify(source)})`));
    assert.ok(await evaluate(`$('compiler-ptx').textContent.includes(${JSON.stringify(ptx)})`));
  }
  assert.ok(await evaluate('$("calculation").getBoundingClientRect().height>0'));
  assert.ok(await evaluate('$("ir").getBoundingClientRect().height>0'));
  await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  assert.ok(await evaluate('document.documentElement.scrollWidth<=390'),'mobile horizontal overflow');
  console.log('PASS: compact laptop layout; all 26 snapshots in all three recording views at 1280×640, 1366×768 and 1440×900; source/PTX switching, inspection and mobile layout');
} finally {
  ws?.close();browser.kill();await Promise.race([exited,sleep(5000)]);rmSync(profile,{recursive:true,force:true,maxRetries:5,retryDelay:200});
}
