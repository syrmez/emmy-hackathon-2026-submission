// Run: node tests/test_softmax.mjs — no packages or GPU required.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const html=readFileSync(new URL('../softmax.html',import.meta.url),'utf8');
const scripts=[...html.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length,2);
for(const [,source] of scripts)new vm.Script(source); // Check UI syntax too.
const context={};
vm.runInNewContext(scripts[0][1],context);
const model=context.SoftmaxDemo;
const sourceIR=readFileSync(new URL('./problems/f02_row_softmax/k_softmax_05263d.ir.txt',import.meta.url),'utf8');
assert.equal(model.IR.trim(),sourceIR.trim(),'Displayed IR must match the real input');
assert.equal(JSON.stringify(model.sample(2026,0)),JSON.stringify(model.sample(2026,0)));
assert.notEqual(JSON.stringify(model.sample(2026,0)),JSON.stringify(model.sample(2026,1)));
assert.throws(()=>model.sample(-1),/Seed/);
assert.throws(()=>model.sample(1,64),/row/);
assert.throws(()=>model.trace([1]),/256/);
assert.throws(()=>model.trace(Array(256).fill(NaN)),/finite/);

function check(input){
  const r=model.trace(input), frames=r.frames, x=r.inputs;
  assert.equal(frames.length,26);
  assert.ok(r.error<1e-6,`softmax max error ${r.error}`);
  assert.ok(Math.abs(r.sum-1)<1e-5,`probability sum ${r.sum}`);
  for(const frame of frames){
    assert.equal(frame.states.length,128);
    assert.equal(frame.shared.length,4);
    for(const pair of [...frame.states,...frame.shared]){
      if(!pair||!pair.ids.length)continue;
      assert.equal(new Set(pair.ids).size,pair.ids.length,'Never count an input twice');
      assert.equal(pair.m,Math.max(...pair.ids.map(i=>x[i])));
      const expected=pair.ids.reduce((sum,i)=>sum+Math.exp(x[i]-pair.m),0);
      assert.ok(Math.abs(pair.s-expected)<1e-4,`incorrect partial summary in ${frame.title}`);
    }
  }
  const shuffles=frames.filter(s=>s.kind==='shuffle');
  assert.equal(shuffles.map(s=>s.offset).join(','),'16,8,4,2,1');
  for(const s of shuffles){
    const before=frames[frames.indexOf(s)-1];
    for(let t=0;t<128;t++){
      assert.equal(t>>5,(t^s.offset)>>5,'Shuffles never cross warps');
      const wanted=model.merge(before.states[t],before.states[t^s.offset]);
      assert.equal(JSON.stringify(s.states[t]),JSON.stringify(wanted),'Use OLD partner state');
    }
  }
  const published=frames.find(s=>s.kind==='publish');
  assert.equal(published.active.join(','),'0,32,64,96');
  assert.ok(published.shared.every(p=>p.ids.length===64));
  assert.equal(published.ready,false);
  const tree=frames.filter(s=>s.kind==='tree');
  assert.equal(tree[0].active.join(','),'0,32');
  assert.equal(tree[1].active.join(','),'0');
  assert.equal(tree[0].shared[0].ids.length,128);
  assert.equal(tree[1].shared[0].ids.length,256);
  for(const s of tree){
    assert.equal(s.ready,false);
    assert.ok(s.states.every((p,t)=>p===published.states[t]),'Tree uses temporary registers');
    const next=frames[frames.indexOf(s)+1];
    assert.equal(next.kind,'barrier');assert.equal(next.active.length,128);assert.equal(next.ready,true);
  }
  const broadcast=frames.find(s=>s.kind==='broadcast');
  assert.ok(broadcast.states.every(p=>p.ids.length===256));
  const stores=frames.filter(s=>s.kind==='store');
  assert.equal(stores[0].output.filter(v=>v!==null).length,128);
  assert.equal(stores[1].output.filter(v=>v!==null).length,256);
  assert.ok(frames[0].output.every(v=>v===null),'Later writes must not mutate earlier snapshots');
  assert.ok(frames[1].states.every(p=>p.s===0&&p.ids.length===0),'Seeds stay immutable');
  assert.ok(published.shared.every(p=>p.ids.length===64),'Tree must not mutate published snapshot');
  assert.ok(r.output.every(v=>Number.isFinite(v)&&v>=0&&v<=1));
  return r;
}
for(const seed of [0,1,2026,4294967295])for(const row of [0,31,63])check(model.sample(seed,row));
for(const constant of [0,1,-1000,1000]){
  const r=check(Array(256).fill(constant));assert.ok(r.output.every(v=>v===1/256));
}
const example=Array(256).fill(0);example[0]=2;example[128]=5;
const local=check(example).frames.filter(s=>s.kind==='local');
assert.equal(local[0].states[0].m,2);assert.equal(local[0].states[0].s,1);
assert.equal(local[1].states[0].m,5);
assert.ok(Math.abs(local[1].states[0].s-(1+Math.exp(-3)))<1e-7);
const spike=Array(256).fill(-1000);spike[173]=1000;
assert.equal(check(spike).output[173],1);
const base=Array.from({length:256},(_,i)=>(i%11)-5);
assert.equal(JSON.stringify(check(base).output),JSON.stringify(check(base.map(v=>v+1000)).output));
console.log('PASS: 20 examples × 26 snapshots; IR, provenance, shuffle simultaneity, tree barriers, immutable history and softmax values.');
