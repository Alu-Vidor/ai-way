import { Species, IrisSample } from '../types';

export function seedShuffle<T>(items: T[], seed: number): T[] {
  const result = [...items];
  let currentIndex = result.length;
  let randomIndex: number;
  let localSeed = seed;

  const random = () => {
    localSeed = Math.sin(localSeed) * 10000;
    return localSeed - Math.floor(localSeed);
  };

  while (currentIndex !== 0) {
    randomIndex = Math.floor(random() * currentIndex);
    currentIndex--;
    [result[currentIndex], result[randomIndex]] = [result[randomIndex], result[currentIndex]];
  }

  return result;
}

export function buildPieData(splits: { train: number; val: number; test: number }) {
  return [
    { name: 'Train', value: splits.train, color: '#6366f1' },
    { name: 'Validation', value: splits.val, color: '#f97316' },
    { name: 'Test', value: splits.test, color: '#22c55e' },
  ];
}

export function formatPercent(value: number | null) {
  if (value === null) return '—';
  return `${Math.round(value * 100)}%`;
}

export function getClassColor(species: Species) {
  switch (species) {
    case 'Setosa':
      return '#facc15';
    case 'Versicolor':
      return '#22c55e';
    case 'Virginica':
      return '#3b82f6';
    default:
      return '#94a3b8';
  }
}

export function toFeatureRow(sample: IrisSample) {
  return [sample.sepalLength, sample.sepalWidth, sample.petalLength, sample.petalWidth];
}
