// Interactive-validation cache toolkit for smap_simulator.jsx (preview app).
// Purpose: let a validation agent read the WHOLE UI state in ONE preview_eval call and drive the app by
// (r,c)/text, instead of many DOM searches (token-efficient + robust). Read-only except the action helpers.
//
// USAGE: paste the IIFE below as a single `preview_eval` expression ONCE per page load (re-install after
// any reload — window globals are cleared on reload). Then call window.__P(), window.__clickCell(r,c), etc.
// Server: preview name "smap", port 5188; get serverId via preview_list. preview_screenshot TIMES OUT —
// always read via the DOM. Do ONE action per eval call, then read window.__P() in a SEPARATE eval
// (synchronous .click() calls batch into one React render).
//
// Bug-toggle buttons render their state as text "SOURCE"/"FIXED" (NOT the label); order in bugRows:
//   [0]=futFilter(Bug A), [1]=kSched(Bug B), [2]=bFaithful(Bug C), [3]=a1a4(proof framing).
// Use window.__src(i) to toggle the i-th by that order. Rectify tabs are buttons "DEF/CAM/DEP/FUT" → __btn('DEP').

(() => {
  window.__grid = () => [...document.querySelectorAll('div')]
    .filter(d => /repeat\(/.test(d.style.gridTemplateColumns || ''))
    .sort((a, b) => b.children.length - a.children.length)[0];
  window.__cell = (r, c) => { const g = window.__grid(); const s = Math.round(Math.sqrt(g.children.length)); return g.children[r * s + c]; };
  window.__clickCell = (r, c) => { window.__cell(r, c).click(); return 'cell ' + r + ',' + c; };
  window.__btn = (t) => { const b = [...document.querySelectorAll('button')].find(b => b.textContent.includes(t)); if (!b) return 'NO BTN:' + t; b.click(); return 'clicked ' + t; };
  window.__src = (i) => { const s = [...document.querySelectorAll('button')].filter(b => { const x = b.textContent.trim(); return x === 'SOURCE' || x === 'FIXED'; }); s[i].click(); return 'toggled #' + i + '=' + s[i].textContent.trim(); };
  window.__mg = () => { const a = document.body.innerText; const m = re => { const x = a.match(re); return x ? x[1] : '?'; }; return {
    f1: m(/align\(F1\)[^\d+\-]*([+\-]?[\d.]+)/), f2: m(/activ\(F2\)[^\d+\-]*([+\-]?[\d.]+)/), mTotal: m(/m TOTAL[^\d+\-]*([+\-]?[\d.]+)/) }; };
  window.__P = () => {
    const a = document.body.innerText; const g = window.__grid(); const side = Math.round(Math.sqrt(g.children.length));
    const grid = [...g.children].map((c, i) => { const t = c.innerText.replace(/\n/g, ' ').trim(); const n = t.match(/[\d.]+/g) || []; return { r: Math.floor(i / side), c: i % side, label: (t.split(' ')[0] || ''), m: parseFloat(n.slice(-1)[0]) }; });
    return {
      step: (a.match(/Step\s+(\d+)/) || [, '?'])[1], rectify: (a.match(/· (DEF|CAM|DEP|FUT)/) || [, '?'])[1],
      showOrig: /show original · ON/.test(a), freeze: /freeze L0 route · ON/.test(a), banner: /VIEWING ORIGINAL/.test(a),
      covered: (a.match(/targets covered\s+\S+/i) || ['?'])[0], selTitle: (a.match(/Point \(\d+,\d+\)/) || ['?'])[0],
      sel: { x: (a.match(/x = ([\d.]+)/) || [, '?'])[1], y: (a.match(/y = ([\d.]+)/) || [, '?'])[1], z: (a.match(/z = ([\d.]+)/) || [, '?'])[1], m: (a.match(/m = ([\d.]+)/) || [, '?'])[1] },
      mTotal: (a.match(/m TOTAL[^\d+\-]*([+\-]?[\d.]+)/) || [, '?'])[1],
      routed: (a.match(/Particle routed[^\n]*/) || ['(none)'])[0].slice(0, 70), nest: (a.match(/L53.63 nest[^\n]*/i) || ['(none)'])[0].slice(0, 160),
      appStandards: (a.match(/[⚠✓][^\n]*standards[^\n]*/i) || ['?'])[0].slice(0, 40),
      activeCells: grid.filter(x => x.m > 0.5).map(x => `(${x.r},${x.c})${x.label}`), fbCells: grid.filter(x => /fb/i.test(x.label)).map(x => `(${x.r},${x.c})`),
      side, gridTop: Math.round(g.getBoundingClientRect().top)
    };
  };
  return 'cache toolkit installed: __P() __clickCell(r,c) __btn(t) __src(i) __mg()';
})()
