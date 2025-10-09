export type Species = 'Setosa' | 'Versicolor' | 'Virginica';

export const speciesLabels: Record<Species, string> = {
  Setosa: 'Сетоза',
  Versicolor: 'Версиколор',
  Virginica: 'Виргиника',
};

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

export type SplitKey = 'train' | 'val' | 'test';

export const splitLabels: Record<SplitKey, string> = {
  train: 'Обучение',
  val: 'Валидация',
  test: 'Тест',
};

export type SplitData = Record<SplitKey, IrisSample[]>;

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
