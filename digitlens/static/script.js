const pad = document.getElementById("pad");
const ctx = pad.getContext("2d");
const seen = document.getElementById("seen");
const seenCtx = seen.getContext("2d");
const digitEl = document.getElementById("digit");
const labelEl = document.getElementById("verdict-label");
const confEl = document.getElementById("confidence");
const barsEl = document.getElementById("bars");
const errorEl = document.getElementById("error");

// ----- build the 10 probability rows once -----
for (let i = 0; i < 10; i++) {
  const li = document.createElement("li");
  li.innerHTML = `<span class="lbl">${i}</span>
    <span class="track"><span class="fill"></span></span>
    <span class="pct">0%</span>`;
  barsEl.appendChild(li);
}
const rows = [...barsEl.children];

// ----- drawing -----
ctx.lineWidth = 30;
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.strokeStyle = "#ffffff";

let drawing = false;
let last = null;
let timer = null;

function pos(e) {
  const r = pad.getBoundingClientRect();
  return {
    x: ((e.clientX - r.left) / r.width) * pad.width,
    y: ((e.clientY - r.top) / r.height) * pad.height,
  };
}

pad.addEventListener("pointerdown", (e) => {
  drawing = true;
  pad.setPointerCapture(e.pointerId);
  last = pos(e);
  ctx.beginPath();
  ctx.arc(last.x, last.y, ctx.lineWidth / 2, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
});

pad.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  const p = pos(e);
  ctx.beginPath();
  ctx.moveTo(last.x, last.y);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  last = p;
  schedulePredict();
});

["pointerup", "pointercancel"].forEach((ev) =>
  pad.addEventListener(ev, () => {
    if (!drawing) return;
    drawing = false;
    schedulePredict(0);
  })
);

document.getElementById("clear").addEventListener("click", clearAll);

function clearAll() {
  ctx.clearRect(0, 0, pad.width, pad.height);
  seenCtx.clearRect(0, 0, 28, 28);
  digitEl.textContent = "–";
  digitEl.classList.add("empty");
  labelEl.textContent = "Waiting for a drawing";
  confEl.textContent = "";
  errorEl.hidden = true;
  rows.forEach((r) => {
    r.classList.remove("top");
    r.querySelector(".fill").style.width = "0%";
    r.querySelector(".pct").textContent = "0%";
  });
}

// ----- prediction (debounced so we don't spam the server) -----
function schedulePredict(delay = 150) {
  clearTimeout(timer);
  timer = setTimeout(predict, delay);
}

async function predict() {
  // The canvas is transparent where nothing is drawn; the server composites it onto black.
  const image = pad.toDataURL("image/png");
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed.");
    errorEl.hidden = true;
    render(data);
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.hidden = false;
  }
}

function render(d) {
  digitEl.textContent = d.digit;
  digitEl.classList.remove("empty");
  labelEl.textContent = `Looks like a ${d.digit}`;
  confEl.textContent = `${(d.confidence * 100).toFixed(1)}% confident`;

  rows.forEach((row, i) => {
    const pct = d.probabilities[i] * 100;
    row.classList.toggle("top", i === d.digit);
    row.querySelector(".fill").style.width = pct + "%";
    row.querySelector(".pct").textContent = pct < 0.1 ? "<0.1%" : pct.toFixed(1) + "%";
  });

  // Draw the 28x28 image the network actually received.
  const img = seenCtx.createImageData(28, 28);
  d.processed.forEach((v, i) => {
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  });
  seenCtx.putImageData(img, 0, 0);
}

clearAll();
