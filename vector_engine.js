// DocuVector AI — High-Dimensional Vector Embedding & Semantic Cosine Retrieval Engine
// Developed by Krishna Gupta

const corpusInput = document.getElementById('corpusInput');
const queryInput = document.getElementById('queryInput');
const corpusStats = document.getElementById('corpusStats');
const btnIndex = document.getElementById('btnIndex');
const btnSearch = document.getElementById('btnSearch');
const resultsList = document.getElementById('resultsList');
const matchStats = document.getElementById('matchStats');

// Telemetry
const telemetryBadge = document.getElementById('telemetryBadge');
const vocabSize = document.getElementById('vocabSize');
const totalChunks = document.getElementById('totalChunks');
const topSimilarity = document.getElementById('topSimilarity');
const calcTime = document.getElementById('calcTime');

// Canvas
const canvas = document.getElementById('vectorCanvas');
const ctx = canvas.getContext('2d');

// Sample Corpora
const SAMPLES = {
  ai: `Deep learning models, particularly Transformers and Vision Transformers (ViT), rely on scaled dot-product self-attention mechanisms to capture long-range token dependencies.

Inference latency scaling is heavily influenced by model parameter quantization (e.g., INT8 and FP4 post-training quantization). By compressing matrix weights, memory bandwidth saturation is reduced on edge hardware.

Safety alignment techniques such as Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) constrain autonomous agents from generating toxic or adversarial completions.

Distributed training pipelines utilize data parallelism, tensor parallelism, and pipeline parallelism to scale 100B+ parameter architectures across multi-node GPU clusters.

Vector databases like Qdrant and Milvus index embeddings using Hierarchical Navigable Small World (HNSW) graphs, achieving sub-millisecond nearest neighbor retrieval at billions-scale.`,

  medical: `Clinical evaluation of patient cohort Alpha-9 demonstrated significant reduction in cardiovascular biomarkers following targeted monoclonal antibody administration.

Pharmacokinetic analysis revealed an elimination half-life of 24.3 hours with steady-state plasma concentrations attained within 72 hours of initial dosage.

Adverse effects were monitored continuously; minor transient hypertension was reported in 4.2% of subjects, resolving spontaneously without secondary clinical intervention.

Genetic sequencing confirmed down-regulation of pro-inflammatory cytokines IL-6 and TNF-alpha, aligning with hypothetical biological therapeutic pathways.`,

  cloud: `Distributed key-value stores leverage consistent hashing rings to partition data across heterogeneous storage nodes while minimizing key reassignment during topology scaling.

Raft consensus algorithms ensure linearizable state machine replication across a five-node quorum even during intermittent network partitions and leader crashes.

Kubernetes container orchestration automates horizontal pod autoscaling based on CPU utilization and Prometheus PromQL metric thresholds.

Zero-trust network architecture mandates mutual TLS (mTLS) authentication and fine-grained SPIFFE/SPIRE cryptographic workload identity attestation.`
};

const STOPWORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
  'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
  'will', 'with', 'or', 'this', 'we', 'they', 'i', 'you', 'can', 'such'
]);

// Engine State
let documentChunks = [];
let vocabulary = [];
let chunkVectors = [];
let idfWeights = {};

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 2 && !STOPWORDS.has(t));
}

// 1. Ingest & Chunk Document
function buildVectorSpace() {
  const text = corpusInput.value.trim();
  if (!text) {
    documentChunks = [];
    vocabulary = [];
    chunkVectors = [];
    updateCorpusStats();
    drawVectorSpace([]);
    return;
  }

  const startTime = performance.now();

  // Split by double line breaks or sentences
  const rawParagraphs = text.split(/\n\s*\n/).filter(p => p.trim().length > 0);
  documentChunks = rawParagraphs.map((p, i) => ({
    id: i + 1,
    text: p.trim(),
    tokens: tokenize(p)
  }));

  // Build Unified Vocabulary
  const vocabMap = new Map();
  const docFreq = {};

  documentChunks.forEach(chunk => {
    const seenInChunk = new Set();
    chunk.tokens.forEach(token => {
      vocabMap.set(token, (vocabMap.get(token) || 0) + 1);
      if (!seenInChunk.has(token)) {
        docFreq[token] = (docFreq[token] || 0) + 1;
        seenInChunk.add(token);
      }
    });
  });

  vocabulary = Array.from(vocabMap.keys());
  const numDocs = documentChunks.length;

  // Calculate Inverse Document Frequency (IDF)
  idfWeights = {};
  vocabulary.forEach(term => {
    const df = docFreq[term] || 1;
    idfWeights[term] = Math.log((numDocs + 1) / (df + 0.5)) + 1;
  });

  // Calculate TF-IDF Vector for each chunk
  chunkVectors = documentChunks.map(chunk => {
    return vectorize(chunk.tokens);
  });

  const duration = Math.round(performance.now() - startTime);

  updateCorpusStats();
  vocabSize.textContent = `${vocabulary.length} dims`;
  totalChunks.textContent = `${documentChunks.length}`;
  telemetryBadge.textContent = `DIMENSIONS: ${vocabulary.length}D`;
  calcTime.textContent = `${duration} ms`;

  // Draw initial space
  drawVectorSpace([]);
}

// Compute normalized TF-IDF vector
function vectorize(tokens) {
  const counts = {};
  tokens.forEach(t => { counts[t] = (counts[t] || 0) + 1; });

  const vector = new Float32Array(vocabulary.length);
  let norm = 0;

  for (let i = 0; i < vocabulary.length; i++) {
    const term = vocabulary[i];
    if (counts[term]) {
      const tf = Math.log(1 + counts[term]);
      const idf = idfWeights[term] || 1;
      const weight = tf * idf;
      vector[i] = weight;
      norm += weight * weight;
    }
  }

  // L2 Normalize
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < vector.length; i++) {
      vector[i] /= norm;
    }
  }

  return vector;
}

// Cosine Similarity between 2 unit vectors (Dot Product)
function cosineSimilarity(vecA, vecB) {
  let dot = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
  }
  return dot;
}

// Perform Semantic Query Search
function executeSemanticSearch() {
  const query = queryInput.value.trim();
  if (!query || chunkVectors.length === 0) return;

  const startTime = performance.now();
  const queryTokens = tokenize(query);
  const queryVector = vectorize(queryTokens);

  // Compute similarities
  const matches = documentChunks.map((chunk, idx) => {
    const score = cosineSimilarity(queryVector, chunkVectors[idx]);
    return {
      chunk,
      score,
      vector: chunkVectors[idx]
    };
  });

  // Sort descending
  matches.sort((a, b) => b.score - a.score);

  const duration = (performance.now() - startTime).toFixed(2);
  calcTime.textContent = `${duration} ms`;

  if (matches.length > 0 && matches[0].score > 0) {
    topSimilarity.textContent = matches[0].score.toFixed(4);
  } else {
    topSimilarity.textContent = '0.0000';
  }

  renderResults(matches, queryTokens);
  drawVectorSpace(matches, queryVector);
}

// Highlight matched query tokens in snippet
function highlightText(text, tokens) {
  let out = text;
  tokens.forEach(token => {
    const regex = new RegExp(`\\b(${token}[a-z]*)\\b`, 'gi');
    out = out.replace(regex, '<mark>$1</mark>');
  });
  return out;
}

function renderResults(matches, queryTokens) {
  const validMatches = matches.filter(m => m.score > 0);
  matchStats.textContent = `${validMatches.length} matches found`;

  if (validMatches.length === 0) {
    resultsList.innerHTML = `
      <div class="empty-state">
        <p>No semantic overlap detected for this concept. Try terms related to the corpus vocabulary.</p>
      </div>
    `;
    return;
  }

  resultsList.innerHTML = validMatches.map((m, rank) => {
    const isTop = rank === 0 && m.score > 0.05;
    const scorePercent = Math.round(m.score * 100);
    return `
      <div class="result-card ${isTop ? 'top-match' : ''}">
        <div class="result-card-header">
          <span class="result-index">Chunk #${m.chunk.id} &bull; Rank ${rank + 1}</span>
          <span class="similarity-badge">Cosine: ${m.score.toFixed(4)} (${scorePercent}%)</span>
        </div>
        <p class="result-snippet">${highlightText(m.chunk.text, queryTokens)}</p>
      </div>
    `;
  }).join('');
}

// 2D Vector Projection & Canvas Radar Rendering
function drawVectorSpace(matches = [], queryVec = null) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;

  // Draw Grid Lines & Concentric Distance Rings
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
  ctx.lineWidth = 1;

  for (let r = 40; r < 200; r += 40) {
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
  }

  ctx.beginPath();
  ctx.moveTo(0, cy); ctx.lineTo(canvas.width, cy);
  ctx.moveTo(cx, 0); ctx.lineTo(cx, canvas.height);
  ctx.stroke();

  if (chunkVectors.length === 0) {
    ctx.fillStyle = '#64748b';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Corpus Empty — Ingest document to map vector coordinates', cx, cy);
    return;
  }

  // 2D Projection: Project high-dimensional vectors to 2D
  // Using pseudo-PCA projection vectors
  const coords = chunkVectors.map((vec, idx) => {
    let x = 0, y = 0;
    for (let i = 0; i < vec.length; i++) {
      const angle = (i * 2.39996); // Golden angle distribution
      x += vec[i] * Math.cos(angle);
      y += vec[i] * Math.sin(angle);
    }
    return {
      x: cx + x * 140,
      y: cy + y * 140,
      id: idx + 1
    };
  });

  // Draw Document Points
  coords.forEach((pt, i) => {
    const isTop = matches.length > 0 && matches[0].chunk.id === pt.id && matches[0].score > 0.05;
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, isTop ? 8 : 5, 0, Math.PI * 2);
    ctx.fillStyle = isTop ? '#10b981' : '#8b5cf6';
    ctx.shadowColor = isTop ? '#10b981' : 'transparent';
    ctx.shadowBlur = isTop ? 15 : 0;
    ctx.fill();

    // Label
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px Fira Code';
    ctx.textAlign = 'center';
    ctx.fillText(`C${pt.id}`, pt.x, pt.y - 9);
  });

  // Draw Query Vector Point & Connection Vector Lines
  if (queryVec) {
    let qx = 0, qy = 0;
    for (let i = 0; i < queryVec.length; i++) {
      const angle = (i * 2.39996);
      qx += queryVec[i] * Math.cos(angle);
      qy += queryVec[i] * Math.sin(angle);
    }
    const queryPoint = { x: cx + qx * 140, y: cy + qy * 140 };

    // Draw connection lines to top matches
    if (matches.length > 0) {
      matches.slice(0, 3).forEach(m => {
        if (m.score > 0) {
          const targetPt = coords[m.chunk.id - 1];
          ctx.beginPath();
          ctx.moveTo(queryPoint.x, queryPoint.y);
          ctx.lineTo(targetPt.x, targetPt.y);
          ctx.strokeStyle = `rgba(16, 185, 129, ${Math.max(0.2, m.score)})`;
          ctx.lineWidth = Math.max(1, m.score * 3);
          ctx.stroke();
        }
      });
    }

    // Draw Query Point
    ctx.beginPath();
    ctx.arc(queryPoint.x, queryPoint.y, 7, 0, Math.PI * 2);
    ctx.fillStyle = '#f59e0b';
    ctx.shadowColor = '#f59e0b';
    ctx.shadowBlur = 12;
    ctx.fill();

    ctx.fillStyle = '#f59e0b';
    ctx.font = 'bold 10px Fira Code';
    ctx.fillText('Query', queryPoint.x, queryPoint.y + 16);
  }

  ctx.shadowBlur = 0; // Reset shadow
}

function updateCorpusStats() {
  const text = corpusInput.value.trim();
  const words = text ? text.split(/\s+/).length : 0;
  corpusStats.textContent = `${words} words • ${documentChunks.length} chunks`;
}

// Event Listeners
btnIndex.addEventListener('click', () => {
  buildVectorSpace();
  if (queryInput.value.trim()) {
    executeSemanticSearch();
  }
});

btnSearch.addEventListener('click', executeSemanticSearch);

queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') executeSemanticSearch();
});

// Sample Buttons
document.querySelectorAll('.sample-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const key = btn.dataset.sample;
    if (SAMPLES[key]) {
      corpusInput.value = SAMPLES[key];
      buildVectorSpace();
      if (key === 'ai') {
        queryInput.value = 'How does quantization affect inference latency?';
        executeSemanticSearch();
      } else if (key === 'medical') {
        queryInput.value = 'What were the blood pressure adverse effects?';
        executeSemanticSearch();
      } else if (key === 'cloud') {
        queryInput.value = 'How does consistent hashing maintain data partitions?';
        executeSemanticSearch();
      }
    }
  });
});

// Query Suggestion Chips
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    queryInput.value = chip.dataset.q;
    executeSemanticSearch();
  });
});

// Initialize with default sample
corpusInput.value = SAMPLES.ai;
buildVectorSpace();
queryInput.value = 'How does quantization affect inference latency?';
executeSemanticSearch();
