---
title: "Transferable Objects"
---

# Transferable Objects

## Introduction

When you send data to a worker with `postMessage()`, the data is **copied** using the structured clone algorithm. For small data, this is fine. But for large data—like images, audio buffers, or large arrays—copying can be slow and memory-intensive.

**Transferable objects** solve this by **transferring** ownership of data instead of copying it. The transfer is nearly instantaneous regardless of data size, but the original thread loses access to the data.

### What We'll Cover

- What transferable objects are
- ArrayBuffer transfer mechanics
- OffscreenCanvas for graphics
- Performance benefits and benchmarks
- Transfer vs copy trade-offs

### Prerequisites

- Understanding of postMessage communication
- Basic knowledge of typed arrays
- Familiarity with binary data in JavaScript

---

## What Are Transferable Objects?

A transferable object is data that can be **moved** from one context (main thread or worker) to another, rather than copied. After transfer, the original reference becomes unusable (zero-length or detached).

```javascript
// Create a large buffer
const buffer = new ArrayBuffer(100_000_000); // 100MB
console.log(buffer.byteLength); // 100000000

// Transfer to worker (near-instant!)
worker.postMessage(buffer, [buffer]);

// Original buffer is now unusable
console.log(buffer.byteLength); // 0 (detached)
```

### Transferable Object Types

| Type | Description | Use Case |
|------|-------------|----------|
| `ArrayBuffer` | Raw binary data | Large datasets, files |
| `MessagePort` | Communication channel | Multi-worker messaging |
| `ImageBitmap` | Bitmap image data | Image processing |
| `OffscreenCanvas` | Offscreen rendering | Graphics in workers |
| `ReadableStream` | Streaming data | Large file processing |
| `WritableStream` | Streaming output | Streaming responses |
| `TransformStream` | Transform pipeline | Data transformation |

---

## ArrayBuffer Transfer

### The Copying Problem

```javascript
// main.js - SLOW: Copying 100MB takes time and doubles memory
const largeArray = new Float32Array(25_000_000); // 100MB

console.time('copy');
worker.postMessage(largeArray); // COPIES the entire array
console.timeEnd('copy'); // ~100-200ms on typical hardware
```

### The Transfer Solution

```javascript
// main.js - FAST: Transfer takes microseconds
const largeArray = new Float32Array(25_000_000); // 100MB

console.time('transfer');
worker.postMessage(largeArray, [largeArray.buffer]); // TRANSFERS the buffer
console.timeEnd('transfer'); // ~0.1ms (nearly instant)

// ⚠️ largeArray is now unusable
console.log(largeArray.length); // 0
console.log(largeArray.buffer.byteLength); // 0
```

### Transfer Syntax

The second argument to `postMessage()` is an array of transferable objects:

```javascript
// Transfer a single buffer
worker.postMessage(data, [data.buffer]);

// Transfer multiple buffers
worker.postMessage(
  { image: imageBuffer, audio: audioBuffer },
  [imageBuffer, audioBuffer]
);

// Transfer with structured data
const message = {
  type: 'process',
  pixels: new Uint8ClampedArray(imageData.data),
  width: 1920,
  height: 1080
};
worker.postMessage(message, [message.pixels.buffer]);
```

### Working with Transferred Data in Workers

```javascript
// worker.js
self.onmessage = (event) => {
  const pixels = event.data.pixels;
  
  // Process the data
  for (let i = 0; i < pixels.length; i += 4) {
    // Invert colors
    pixels[i] = 255 - pixels[i];       // R
    pixels[i + 1] = 255 - pixels[i + 1]; // G
    pixels[i + 2] = 255 - pixels[i + 2]; // B
    // Alpha unchanged
  }
  
  // Transfer back to main thread
  self.postMessage({ pixels }, [pixels.buffer]);
};
```

---

## OffscreenCanvas

`OffscreenCanvas` enables canvas rendering in workers, freeing the main thread from graphics work.

### Creating OffscreenCanvas

```javascript
// Method 1: Transfer from existing canvas
const canvas = document.getElementById('myCanvas');
const offscreen = canvas.transferControlToOffscreen();
worker.postMessage({ canvas: offscreen }, [offscreen]);

// Method 2: Create directly in worker (no visible canvas)
// In worker.js:
const offscreen = new OffscreenCanvas(1920, 1080);
const ctx = offscreen.getContext('2d');
```

### Rendering in a Worker

```javascript
// main.js
const canvas = document.getElementById('gameCanvas');
const offscreen = canvas.transferControlToOffscreen();

const worker = new Worker('render-worker.js');
worker.postMessage({ canvas: offscreen, width: 800, height: 600 }, [offscreen]);

// Animation loop runs in worker - main thread stays responsive!
```

```javascript
// render-worker.js
let canvas, ctx;
let animationId;

self.onmessage = (event) => {
  const { canvas: offscreen, width, height } = event.data;
  canvas = offscreen;
  canvas.width = width;
  canvas.height = height;
  ctx = canvas.getContext('2d');
  
  animate();
};

function animate() {
  // Clear
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  // Draw something (e.g., particles, game graphics)
  ctx.fillStyle = '#0f0';
  const x = (Date.now() / 10) % canvas.width;
  ctx.fillRect(x, 100, 50, 50);
  
  // Schedule next frame
  animationId = requestAnimationFrame(animate);
}
```

### Image Processing with OffscreenCanvas

```javascript
// worker.js - Process image without blocking main thread
self.onmessage = async (event) => {
  const { imageData, filter } = event.data;
  
  // Create offscreen canvas
  const canvas = new OffscreenCanvas(imageData.width, imageData.height);
  const ctx = canvas.getContext('2d');
  
  // Put image data
  ctx.putImageData(imageData, 0, 0);
  
  // Apply filter using canvas operations
  if (filter === 'blur') {
    ctx.filter = 'blur(5px)';
    ctx.drawImage(canvas, 0, 0);
  }
  
  // Get processed image
  const blob = await canvas.convertToBlob({ type: 'image/png' });
  self.postMessage({ blob });
};
```

---

## Performance Benefits

### Benchmark: Copy vs Transfer

```javascript
// Benchmark function
async function benchmark(size) {
  const data = new Float32Array(size);
  
  // Fill with random data
  for (let i = 0; i < size; i++) {
    data[i] = Math.random();
  }
  
  const copyWorker = new Worker('worker.js');
  const transferWorker = new Worker('worker.js');
  
  // Benchmark copy
  const copyStart = performance.now();
  await new Promise(resolve => {
    copyWorker.onmessage = resolve;
    copyWorker.postMessage(data); // Copy
  });
  const copyTime = performance.now() - copyStart;
  
  // Benchmark transfer
  const data2 = new Float32Array(size);
  for (let i = 0; i < size; i++) data2[i] = Math.random();
  
  const transferStart = performance.now();
  await new Promise(resolve => {
    transferWorker.onmessage = resolve;
    transferWorker.postMessage(data2, [data2.buffer]); // Transfer
  });
  const transferTime = performance.now() - transferStart;
  
  return { copyTime, transferTime };
}

// Results (approximate, varies by hardware):
// 1MB:   Copy: ~5ms,    Transfer: ~0.1ms
// 10MB:  Copy: ~40ms,   Transfer: ~0.1ms
// 100MB: Copy: ~400ms,  Transfer: ~0.1ms
```

### Real-World Performance Table

| Data Size | Copy Time | Transfer Time | Speedup |
|-----------|-----------|---------------|---------|
| 1 MB | ~5 ms | ~0.1 ms | 50x |
| 10 MB | ~40 ms | ~0.1 ms | 400x |
| 100 MB | ~400 ms | ~0.1 ms | 4000x |
| 1 GB | ~4000 ms | ~0.1 ms | 40000x |

> **Note:** Transfer time is constant regardless of data size!

### Memory Efficiency

```javascript
// Copy: Uses 2x memory temporarily
const buffer = new ArrayBuffer(100_000_000); // 100MB allocated
worker.postMessage(buffer); // Another 100MB allocated during copy
// Peak: 200MB, then original can be GC'd

// Transfer: Uses 1x memory always
const buffer = new ArrayBuffer(100_000_000); // 100MB allocated
worker.postMessage(buffer, [buffer]); // Same 100MB moved
// Peak: 100MB (no duplication)
```

---

## Transfer vs Copy Trade-offs

### When to Use Transfer

| Scenario | Recommendation |
|----------|----------------|
| Data > 1MB | ✅ Transfer |
| One-way data flow | ✅ Transfer |
| Original data not needed after send | ✅ Transfer |
| Performance critical | ✅ Transfer |
| Image/audio processing | ✅ Transfer |

### When to Use Copy

| Scenario | Recommendation |
|----------|----------------|
| Data < 10KB | 📋 Copy (overhead not worth it) |
| Need original data after send | 📋 Copy |
| Sending to multiple workers | 📋 Copy (or clone before transfer) |
| Immutable data patterns | 📋 Copy |

### Handling "Need Data in Both Places"

```javascript
// Option 1: Clone before transfer
const original = new Float32Array([1, 2, 3, 4]);
const copy = new Float32Array(original); // Create copy first
worker.postMessage(original, [original.buffer]); // Transfer original
console.log(copy); // Still have the copy

// Option 2: Transfer back and forth
// main.js
worker.postMessage(data, [data.buffer]);
worker.onmessage = (e) => {
  data = e.data; // Received transferred data back
};

// worker.js
self.onmessage = (e) => {
  const data = e.data;
  // Process...
  self.postMessage(data, [data.buffer]); // Transfer back
};
```

---

## Streaming with TransformStream

For very large data, use streams to process chunks:

```javascript
// main.js
const { readable, writable } = new TransformStream();

worker.postMessage(
  { readable, writable },
  [readable, writable]
);

// Write data in chunks
const writer = writable.getWriter();
for (const chunk of largeDataChunks) {
  await writer.write(chunk);
}
await writer.close();
```

```javascript
// worker.js
self.onmessage = async (event) => {
  const { readable, writable } = event.data;
  
  const reader = readable.getReader();
  const writer = writable.getWriter();
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    // Process chunk
    const processed = value.map(x => x * 2);
    await writer.write(processed);
  }
  
  await writer.close();
};
```

---

## Common Patterns

### Round-Trip Transfer

```javascript
// main.js
async function processInWorker(data) {
  return new Promise((resolve) => {
    const worker = new Worker('processor.js');
    
    worker.onmessage = (e) => {
      resolve(e.data.buffer);
      worker.terminate();
    };
    
    // Send AND receive with transfer
    worker.postMessage(data, [data.buffer]);
  });
}

// worker.js
self.onmessage = (event) => {
  const data = event.data;
  
  // Process in place (modifying the transferred buffer)
  for (let i = 0; i < data.length; i++) {
    data[i] = data[i] * 2;
  }
  
  // Transfer back
  self.postMessage(data, [data.buffer]);
};
```

### Image Processing Pipeline

```javascript
// main.js
async function processImage(imageElement) {
  // Create canvas and get image data
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = imageElement.width;
  canvas.height = imageElement.height;
  ctx.drawImage(imageElement, 0, 0);
  
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  
  // Transfer to worker
  const worker = new Worker('image-worker.js');
  
  return new Promise((resolve) => {
    worker.onmessage = (e) => {
      const processedData = new ImageData(
        new Uint8ClampedArray(e.data.pixels),
        e.data.width,
        e.data.height
      );
      ctx.putImageData(processedData, 0, 0);
      resolve(canvas.toDataURL());
      worker.terminate();
    };
    
    worker.postMessage({
      pixels: imageData.data,
      width: imageData.width,
      height: imageData.height
    }, [imageData.data.buffer]);
  });
}
```

---

## Hands-on Exercise

### Your Task

Create an image filter worker that applies filters using transferred data.

### Requirements

1. Load an image from URL
2. Transfer pixel data to a worker
3. Apply a grayscale filter in the worker
4. Transfer the result back
5. Display the processed image

### Expected Result

Original image → Grayscale version, with transfer time < 1ms for any image size.

<details>
<summary>💡 Hints</summary>

- Use `createImageBitmap()` and `OffscreenCanvas` for efficient image handling
- Transfer `imageData.data.buffer`, not just `imageData`
- Remember to reconstruct `ImageData` after transfer

</details>

<details>
<summary>✅ Solution</summary>

**main.js:**
```javascript
async function processImage(url) {
  // Load image
  const img = new Image();
  img.crossOrigin = 'anonymous';
  await new Promise((resolve) => {
    img.onload = resolve;
    img.src = url;
  });
  
  // Create canvas and get image data
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);
  
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  
  // Transfer to worker
  const worker = new Worker('filter-worker.js');
  
  const startTime = performance.now();
  
  const result = await new Promise((resolve) => {
    worker.onmessage = (e) => {
      console.log('Transfer time:', performance.now() - startTime, 'ms');
      resolve(e.data);
    };
    
    worker.postMessage({
      pixels: imageData.data,
      width: imageData.width,
      height: imageData.height,
      filter: 'grayscale'
    }, [imageData.data.buffer]);
  });
  
  // Apply result
  const processedData = new ImageData(
    new Uint8ClampedArray(result.pixels),
    result.width,
    result.height
  );
  ctx.putImageData(processedData, 0, 0);
  
  document.body.appendChild(canvas);
  worker.terminate();
}

processImage('https://picsum.photos/1920/1080');
```

**filter-worker.js:**
```javascript
self.onmessage = (event) => {
  const { pixels, width, height, filter } = event.data;
  
  if (filter === 'grayscale') {
    for (let i = 0; i < pixels.length; i += 4) {
      const avg = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
      pixels[i] = avg;     // R
      pixels[i + 1] = avg; // G
      pixels[i + 2] = avg; // B
      // Alpha unchanged
    }
  }
  
  self.postMessage({ pixels, width, height }, [pixels.buffer]);
};
```

</details>

---

## Summary

✅ Transferable objects **move** data instead of copying
✅ Transfer is near-instant regardless of data size
✅ Original reference becomes unusable after transfer
✅ Use for `ArrayBuffer`, `ImageBitmap`, `OffscreenCanvas`, streams
✅ Syntax: `postMessage(data, [transferables])`
✅ Essential for performance with large data (images, audio, ML models)

**Next:** [Shared Workers](./04-shared-workers.md)

---

## Further Reading

- [MDN Transferable Objects](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferable_objects) - Complete reference
- [MDN OffscreenCanvas](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas) - Graphics in workers
- [Transferables Explained](https://developer.chrome.com/blog/transferable-objects-lightning-fast/) - Chrome blog deep dive

<!-- 
Sources Consulted:
- MDN Transferable Objects: https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferable_objects
- MDN OffscreenCanvas: https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas
-->
