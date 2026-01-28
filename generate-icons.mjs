import sharp from 'sharp';
import { mkdir } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const sizes = [72, 96, 128, 144, 152, 192, 384, 512];

// Create a simple AI icon using sharp
async function generateIcons() {
  const iconsDir = join(__dirname, 'public/icons');
  
  // Ensure icons directory exists
  await mkdir(iconsDir, { recursive: true });
  
  for (const size of sizes) {
    // Create a gradient background with text
    const svg = `
      <svg width="${size}" height="${size}" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#6366f1;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#8b5cf6;stop-opacity:1" />
          </linearGradient>
        </defs>
        <rect width="${size}" height="${size}" rx="${size * 0.125}" fill="url(#grad)"/>
        <g fill="white">
          <!-- Central brain/AI icon -->
          <circle cx="${size * 0.5}" cy="${size * 0.38}" r="${size * 0.08}" />
          <circle cx="${size * 0.35}" cy="${size * 0.52}" r="${size * 0.06}" />
          <circle cx="${size * 0.65}" cy="${size * 0.52}" r="${size * 0.06}" />
          <circle cx="${size * 0.42}" cy="${size * 0.65}" r="${size * 0.045}" />
          <circle cx="${size * 0.58}" cy="${size * 0.65}" r="${size * 0.045}" />
          <!-- Neural connections -->
          <line x1="${size * 0.5}" y1="${size * 0.46}" x2="${size * 0.35}" y2="${size * 0.46}" stroke="white" stroke-width="${Math.max(2, size * 0.015)}"/>
          <line x1="${size * 0.5}" y1="${size * 0.46}" x2="${size * 0.65}" y2="${size * 0.46}" stroke="white" stroke-width="${Math.max(2, size * 0.015)}"/>
          <line x1="${size * 0.35}" y1="${size * 0.58}" x2="${size * 0.42}" y2="${size * 0.605}" stroke="white" stroke-width="${Math.max(1.5, size * 0.012)}"/>
          <line x1="${size * 0.65}" y1="${size * 0.58}" x2="${size * 0.58}" y2="${size * 0.605}" stroke="white" stroke-width="${Math.max(1.5, size * 0.012)}"/>
        </g>
        <text x="${size * 0.5}" y="${size * 0.88}" font-family="Arial, sans-serif" font-size="${size * 0.12}" font-weight="bold" fill="white" text-anchor="middle">AI</text>
      </svg>
    `;
    
    const outputPath = join(iconsDir, `icon-${size}x${size}.png`);
    
    await sharp(Buffer.from(svg))
      .png()
      .toFile(outputPath);
    
    console.log(`Generated: icon-${size}x${size}.png`);
  }
  
  console.log('\nAll PWA icons generated successfully!');
}

generateIcons().catch(console.error);
