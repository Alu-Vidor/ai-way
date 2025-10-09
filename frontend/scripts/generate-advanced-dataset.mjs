import { writeFileSync } from 'fs';

const samples = [];
const classes = [
  { id: 'Orion', offset: 0 },
  { id: 'Andromeda', offset: 2.1 },
  { id: 'Centaurus', offset: 4.2 },
];

let seed = 2024;
function random() {
  seed = Math.sin(seed) * 10000;
  return seed - Math.floor(seed);
}

function noise(scale) {
  return (random() - 0.5) * 2 * scale;
}

const perClass = 220;

classes.forEach((cls, classIndex) => {
  for (let i = 0; i < perClass; i++) {
    const baseAngle = (i / perClass) * Math.PI * 2 + cls.offset;
    const radius = 1.3 + classIndex * 0.8 + noise(0.35);
    const angle = baseAngle + noise(0.25);
    const x = radius * Math.cos(angle) + noise(0.4);
    const y = radius * Math.sin(angle) + noise(0.4);
    const orbitalSpeed = radius * (1.2 + noise(0.2)) + classIndex * 0.4;
    const luminosity = Math.sin(angle * 2 + classIndex * 0.3) + classIndex * 0.8 + noise(0.5);
    const turbulence = Math.cos(angle * 1.5) * 0.7 + classIndex * 0.5 + noise(0.6);

    samples.push({
      id: samples.length,
      orbitRadius: Number(radius.toFixed(3)),
      orbitalSpeed: Number(orbitalSpeed.toFixed(3)),
      luminosity: Number(luminosity.toFixed(3)),
      turbulence: Number(turbulence.toFixed(3)),
      pcaX: Number(x.toFixed(3)),
      pcaY: Number(y.toFixed(3)),
      pattern: cls.id,
    });
  }
});

writeFileSync(new URL('../src/assets/advanced.json', import.meta.url), JSON.stringify(samples, null, 2));
