export type Species = 'Setosa' | 'Versicolor' | 'Virginica';

export interface IrisSample {
  id: number;
  sepalLength: number;
  sepalWidth: number;
  petalLength: number;
  petalWidth: number;
  pcaX: number;
  pcaY: number;
  species: Species;
}

export interface LayerConfig {
  id: string;
  units: number;
  activation: 'relu' | 'sigmoid' | 'softmax';
  dropout: number;
  batchNorm: boolean;
}

export type SplitData = Record<'train' | 'val' | 'test', IrisSample[]>;

export interface HistoryPoint {
  epoch: number;
  value: number;
}

export interface ConfusionMatrixCell {
  actual: Species;
  predicted: Species;
  value: number;
}

export interface FeatureStats {
  means: number[];
  std: number[];
}

export interface SamplePredictionRow {
  sample: IrisSample;
  predicted: Species;
  correct: boolean;
}

export interface TrainingSummary {
  history: HistoryPoint[];
  accuracyHistory: HistoryPoint[];
  valHistory: HistoryPoint[];
  valAccuracyHistory: HistoryPoint[];
  testAccuracy: number | null;
  confusionMatrix: ConfusionMatrixCell[][] | null;
  samplePredictions: SamplePredictionRow[];
  featureStats: FeatureStats | null;
}

export interface DecisionGridData {
  grid: number[][];
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}
