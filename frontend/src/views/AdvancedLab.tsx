import { useCallback, useMemo, useState } from 'react';
import { DndContext, closestCenter, KeyboardSensor, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import type { DragEndEvent } from '@dnd-kit/core';
import { arrayMove, SortableContext, sortableKeyboardCoordinates, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { v4 as uuid } from 'uuid';
import * as tf from '@tensorflow/tfjs';
import advancedData from '../assets/advanced.json';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import { Label } from '../components/ui/label';
import { Slider } from '../components/ui/slider';
import { DecisionBoundaryCanvas } from '../components/DecisionBoundaryCanvas';
import { ConfusionMatrixTable } from '../components/ConfusionMatrixTable';
import {
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
} from 'recharts';
import { SortableLayerCard } from '../components/SortableLayerCard';
import { NetworkOutputCard } from '../components/NetworkOutputCard';
import { ModelHints } from '../components/ModelHints';
import { buildPieData, formatPercent, seedShuffle } from '../lib/utils';
import { DecisionGridData, FeatureStats, LayerConfig, HistoryPoint, SplitKey, splitLabels } from '../types';

const patternOrder: GalaxyClass[] = ['Orion', 'Andromeda', 'Centaurus'];
const splitKeys: SplitKey[] = ['train', 'val', 'test'];

const defaultLayers: LayerConfig[] = [
  {
    id: uuid(),
    units: 10,
    activation: 'relu',
    dropout: 0.1,
    batchNorm: false,
  },
];

const minSplit = 10;
const minEpochs = 5;
const maxEpochs = 30;
const defaultEpochs = 18;

function useLayerSensors() {
  return useSensors(
    useSensor(PointerSensor, {
      activationConstraint: { distance: 6 },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );
}

function useSplitState() {
  const [splits, setSplits] = useState({ train: 70, val: 15, test: 15 });

  const updateTrain = useCallback(
    (value: number) => {
      let nextTrain = Math.round(value);
      const maxTrain = 100 - minSplit * 2;
      if (nextTrain > maxTrain) nextTrain = maxTrain;
      if (nextTrain < minSplit) nextTrain = minSplit;
      const remaining = 100 - nextTrain;
      let nextVal = splits.val;
      if (remaining - nextVal < minSplit) {
        nextVal = remaining - minSplit;
      }
      if (nextVal < minSplit) nextVal = minSplit;
      const nextTest = 100 - nextTrain - nextVal;
      setSplits({ train: nextTrain, val: nextVal, test: nextTest });
    },
    [splits.val]
  );

  const updateVal = useCallback(
    (value: number) => {
      let nextVal = Math.round(value);
      if (nextVal < minSplit) nextVal = minSplit;
      if (nextVal > 100 - minSplit * 2) nextVal = 100 - minSplit * 2;
      let nextTrain = splits.train;
      if (nextTrain + nextVal > 100 - minSplit) {
        nextTrain = 100 - minSplit - nextVal;
      }
      if (nextTrain < minSplit) nextTrain = minSplit;
      const nextTest = 100 - nextTrain - nextVal;
      setSplits({ train: nextTrain, val: nextVal, test: nextTest });
    },
    [splits.train]
  );

  return { splits, updateTrain, updateVal };
}

const initialState: AdvancedTrainingSummary = {
  history: [],
  accuracyHistory: [],
  valHistory: [],
  valAccuracyHistory: [],
  testAccuracy: null,
  confusionMatrix: null,
  samplePredictions: [],
  featureStats: null,
};

export function AdvancedLab() {
  const layerSensors = useLayerSensors();
  const { splits, updateTrain, updateVal } = useSplitState();
  const [seed, setSeed] = useState(1337);
  const [dataReady, setDataReady] = useState(false);
  const [dataSplit, setDataSplit] = useState<GalaxySplitData | null>(null);
  const [layers, setLayers] = useState<LayerConfig[]>(defaultLayers);
  const [isTraining, setIsTraining] = useState(false);
  const [summary, setSummary] = useState<AdvancedTrainingSummary>(initialState);
  const [decisionGrid, setDecisionGrid] = useState<DecisionGridData<GalaxyClass> | null>(null);
  const [activeScatter, setActiveScatter] = useState<SplitKey>('train');
  const [epochCount, setEpochCount] = useState(defaultEpochs);

  const samples = useMemo(() => {
    return (advancedData as GalaxySample[]).map((item, index) => ({ ...item, id: index }));
  }, []);

  const handleApplyData = useCallback(() => {
    const shuffled = seedShuffle(samples, seed);
    const total = shuffled.length;
    const trainCount = Math.round((splits.train / 100) * total);
    const valCount = Math.round((splits.val / 100) * total);
    const train = shuffled.slice(0, trainCount);
    const val = shuffled.slice(trainCount, trainCount + valCount);
    const test = shuffled.slice(trainCount + valCount);
    setDataSplit({ train, val, test });
    setDataReady(true);
    setSummary(initialState);
    setDecisionGrid(null);
  }, [samples, seed, splits]);

  const handleAddLayer = () => {
    setLayers((prev) => [
      ...prev,
      {
        id: uuid(),
        units: 8,
        activation: 'relu',
        dropout: 0.1,
        batchNorm: false,
      },
    ]);
  };

  const handleLayerChange = useCallback((id: string, changes: Partial<LayerConfig>) => {
    setLayers((prev) => prev.map((layer) => (layer.id === id ? { ...layer, ...changes } : layer)));
  }, []);

  const handleLayerRemove = useCallback((id: string) => {
    setLayers((prev) => prev.filter((layer) => layer.id !== id));
  }, []);

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const { active, over } = event;
      if (!over || active.id === over.id) return;
      setLayers((prev) => {
        const oldIndex = prev.findIndex((layer) => layer.id === active.id);
        const newIndex = prev.findIndex((layer) => layer.id === over.id);
        return arrayMove(prev, oldIndex, newIndex);
      });
    },
    []
  );

  const datasetForScatter = useMemo(() => {
    if (!dataSplit) return [];
    const source = dataSplit[activeScatter];
    return source.map((item) => ({ ...item }));
  }, [dataSplit, activeScatter]);

  const pieData = useMemo(() => buildPieData(splits), [splits]);

  const canTrain = dataReady && dataSplit && dataSplit.train.length > 0 && !isTraining;

  const handleTrain = useCallback(async () => {
    if (!dataSplit || isTraining) return;
    setIsTraining(true);
    setSummary(initialState);
    setDecisionGrid(null);

    const disposeTensors: tf.Tensor[] = [];

    try {
      const labelToIndex: Record<GalaxyClass, number> = {
        Orion: 0,
        Andromeda: 1,
        Centaurus: 2,
      };

      const trainFeatures = dataSplit.train.map(toFeatureRow);
      const valFeatures = dataSplit.val.map(toFeatureRow);
      const testFeatures = dataSplit.test.map(toFeatureRow);

      const featureMeans = trainFeatures[0].map((_, featureIndex) =>
        trainFeatures.reduce((sum, row) => sum + row[featureIndex], 0) / trainFeatures.length
      );
      const featureStd = trainFeatures[0].map((_, featureIndex) => {
        const mean = featureMeans[featureIndex];
        const variance =
          trainFeatures.reduce((sum, row) => sum + (row[featureIndex] - mean) ** 2, 0) / trainFeatures.length;
        return Math.max(Math.sqrt(variance), 1e-6);
      });

      const standardize = (rows: number[][]) =>
        rows.map((row) => row.map((value, index) => (value - featureMeans[index]) / featureStd[index]));

      const toTensor = (rows: number[][]) => {
        const tensor = tf.tensor2d(rows);
        disposeTensors.push(tensor);
        return tensor;
      };

      const encodeLabels = (samples: GalaxySample[]) =>
        samples.map((sample) => {
          const vector = [0, 0, 0];
          vector[labelToIndex[sample.pattern]] = 1;
          return vector;
        });

      const trainXs = toTensor(standardize(trainFeatures));
      const trainYs = toTensor(encodeLabels(dataSplit.train));
      const valXs = dataSplit.val.length ? toTensor(standardize(valFeatures)) : null;
      const valYs = dataSplit.val.length ? toTensor(encodeLabels(dataSplit.val)) : null;
      const testXs = dataSplit.test.length ? toTensor(standardize(testFeatures)) : null;
      const testYs = dataSplit.test.length ? toTensor(encodeLabels(dataSplit.test)) : null;

      const model = tf.sequential();
      layers.forEach((layer, index) => {
        model.add(
          tf.layers.dense({
            units: layer.units,
            activation: layer.activation,
            inputShape: index === 0 ? [4] : undefined,
          })
        );
        if (layer.batchNorm) {
          model.add(tf.layers.batchNormalization());
        }
        if (layer.dropout > 0) {
          model.add(tf.layers.dropout({ rate: layer.dropout }));
        }
      });

      model.add(tf.layers.dense({ units: patternOrder.length, activation: 'softmax' }));

      model.compile({
        optimizer: tf.train.adam(0.005),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
      });

      const historyPoints: HistoryPoint[] = [];
      const accuracyPoints: HistoryPoint[] = [];
      const valHistoryPoints: HistoryPoint[] = [];
      const valAccuracyPoints: HistoryPoint[] = [];

      await model.fit(trainXs, trainYs, {
        epochs: epochCount,
        shuffle: true,
        validationData: valXs && valYs ? [valXs, valYs] : undefined,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            historyPoints.push({ epoch: epoch + 1, value: logs?.loss ?? 0 });
            accuracyPoints.push({ epoch: epoch + 1, value: (logs?.acc ?? logs?.accuracy ?? 0) as number });
            if (logs?.val_loss !== undefined) {
              valHistoryPoints.push({ epoch: epoch + 1, value: logs.val_loss });
            }
            if (logs?.val_acc !== undefined || logs?.val_accuracy !== undefined) {
              valAccuracyPoints.push({
                epoch: epoch + 1,
                value: (logs.val_acc ?? logs.val_accuracy ?? 0) as number,
              });
            }
            setSummary((prev) => ({
              ...prev,
              history: [...historyPoints],
              accuracyHistory: [...accuracyPoints],
              valHistory: [...valHistoryPoints],
              valAccuracyHistory: [...valAccuracyPoints],
            }));
            await tf.nextFrame();
          },
        },
      });

      let testAccuracy: number | null = null;
      if (testXs && testYs) {
        const evaluation = model.evaluate(testXs, testYs) as tf.Scalar[];
        const accTensor = evaluation[1];
        testAccuracy = accTensor?.dataSync()[0] ?? null;
        disposeTensors.push(...evaluation);
      }

      let confusionMatrix: GalaxyConfusionCell[][] | null = null;
      let samplesPreview: AdvancedPrediction[] = [];

      if (testXs && testYs && dataSplit.test.length) {
        const predictions = model.predict(testXs) as tf.Tensor;
        disposeTensors.push(predictions);
        const predArray = predictions.arraySync() as number[][];
        const predictedIndices = predArray.map((row) => row.indexOf(Math.max(...row)));
        const confusion = Array.from({ length: patternOrder.length }, () => Array(patternOrder.length).fill(0));
        dataSplit.test.forEach((sample, index) => {
          const predictedIndex = predictedIndices[index];
          const actualIndex = labelToIndex[sample.pattern];
          confusion[actualIndex][predictedIndex] += 1;
        });
        confusionMatrix = confusion.map((row, actual) =>
          row.map((value, predicted) => ({
            actual: patternOrder[actual],
            predicted: patternOrder[predicted],
            value,
          }))
        );

        const previewSource = dataSplit.test.map((sample, index) => ({ sample, predictedIndex: predictedIndices[index] }));
        const rng = seedShuffle(previewSource, seed).slice(0, 5);
        samplesPreview = rng.map(({ sample, predictedIndex }) => ({
          sample,
          predicted: patternOrder[predictedIndex],
          correct: patternOrder[predictedIndex] === sample.pattern,
        }));
      }

      const decision = generateGalaxyDecisionGrid(model, featureMeans, featureStd, samples);
      setDecisionGrid(decision);

      setSummary({
        history: historyPoints,
        accuracyHistory: accuracyPoints,
        valHistory: valHistoryPoints,
        valAccuracyHistory: valAccuracyPoints,
        testAccuracy,
        confusionMatrix,
        samplePredictions: samplesPreview,
        featureStats: { means: featureMeans, std: featureStd },
      });

      model.dispose();
    } catch (error) {
      console.error(error);
    } finally {
      disposeTensors.forEach((tensor) => tensor.dispose());
      setIsTraining(false);
    }
  }, [dataSplit, epochCount, isTraining, layers, samples, seed]);

  const summaryCards = useMemo(() => {
    if (!summary.history.length) return null;

    const lineData = summary.history.map((point, index) => ({
      epoch: point.epoch,
      loss: point.value,
      valLoss: summary.valHistory[index]?.value ?? null,
    }));

    const accData = summary.accuracyHistory.map((point, index) => ({
      epoch: point.epoch,
      accuracy: point.value,
      valAccuracy: summary.valAccuracyHistory[index]?.value ?? null,
    }));

    return (
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Как уменьшается ошибка" description="Следи, как сеть учится распознавать паттерны галактик по эпохам.">
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="epoch" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} />
                <Tooltip formatter={(value: number) => value.toFixed(3)} labelFormatter={(label) => `Эпоха ${label}`} />
                <Legend />
                <Line type="monotone" dataKey="loss" name="Обучение" stroke="#0ea5e9" strokeWidth={2} dot={false} />
                {summary.valHistory.length > 0 && (
                  <Line
                    type="monotone"
                    dataKey="valLoss"
                    name="Валидация"
                    stroke="#f97316"
                    strokeWidth={2}
                    dot={false}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Как растёт точность" description="Здесь видно, сколько галактик сеть классифицирует правильно.">
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={accData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="epoch" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} domain={[0, 1]} />
                <Tooltip
                  formatter={(value: number) => `${Math.round(value * 100)}%`}
                  labelFormatter={(label) => `Эпоха ${label}`}
                />
                <Legend />
                <Line type="monotone" dataKey="accuracy" name="Обучение" stroke="#22c55e" strokeWidth={2} dot={false} />
                {summary.valAccuracyHistory.length > 0 && (
                  <Line
                    type="monotone"
                    dataKey="valAccuracy"
                    name="Валидация"
                    stroke="#facc15"
                    strokeWidth={2}
                    dot={false}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
    );
  }, [summary]);

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-8 p-6">
      <header className="space-y-4 text-center">
        <h1 className="text-4xl font-black text-primary">Продвинутая миссия: галактики 🪐</h1>
        <p className="mx-auto max-w-3xl text-lg text-muted-foreground">
          Теперь сеть должна отличать сложные космические паттерны. Каждый объект описан орбитой, скоростью вращения,
          светимостью и уровнем турбулентности. Данные куда шумнее, а классы — ближе друг к другу.
        </p>
      </header>

      <section className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
        <Card title="Подготовь данные" description="Раздели космический каталог и загрузь его в лабораторию.">
          <div className="flex flex-col gap-6">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label htmlFor="train">{splitLabels.train}</Label>
                  <span className="text-sm font-semibold text-foreground">{splits.train}%</span>
                </div>
                <Slider
                  id="train"
                  value={[splits.train]}
                  min={minSplit}
                  max={85}
                  step={5}
                  onValueChange={([value]) => updateTrain(value)}
                />
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label htmlFor="val">{splitLabels.val}</Label>
                  <span className="text-sm font-semibold text-foreground">{splits.val}%</span>
                </div>
                <Slider id="val" value={[splits.val]} min={minSplit} max={50} step={5} onValueChange={([value]) => updateVal(value)} />
              </div>
            </div>

            <div className="flex items-center gap-4">
              <Label htmlFor="seed" className="shrink-0">
                Случайное число
              </Label>
              <input
                id="seed"
                type="number"
                value={seed}
                onChange={(event) => setSeed(Number(event.target.value))}
                className="w-28 rounded-lg border border-border px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              />
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="h-44 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie dataKey="value" data={pieData} innerRadius={30} outerRadius={70} paddingAngle={4}>
                      {pieData.map((entry) => (
                        <Cell key={entry.name} fill={entry.color} stroke="white" />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value: number) => `${value}%`} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="flex flex-col justify-center gap-2 text-sm">
                {pieData.map((entry) => (
                  <div key={entry.name} className="flex items-center justify-between rounded-lg border border-border px-3 py-2">
                    <span className="font-medium text-foreground">{entry.name}</span>
                    <span className="font-semibold text-muted-foreground">{entry.value}%</span>
                  </div>
                ))}
              </div>
            </div>

            <Button onClick={handleApplyData}>Загрузить космический датасет</Button>

            {dataSplit && (
              <div className="grid gap-4 md:grid-cols-3">
                {splitKeys.map((key) => (
                  <div key={key} className="rounded-lg border border-border bg-white/70 p-4 text-sm shadow-sm">
                    <p className="font-semibold uppercase text-muted-foreground">{splitLabels[key].toUpperCase()}</p>
                    <p className="text-2xl font-bold text-foreground">{dataSplit[key].length}</p>
                  </div>
                ))}
              </div>
            )}

            {dataSplit && (
              <div className="flex flex-wrap gap-2 text-sm">
                {patternOrder.map((pattern) => (
                  <span
                    key={pattern}
                    className="flex items-center gap-2 rounded-full bg-muted px-3 py-1 font-medium text-muted-foreground"
                  >
                    <span className="h-3 w-3 rounded-full" style={{ background: getPatternColor(pattern) }} />
                    {patternLabels[pattern]}
                  </span>
                ))}
              </div>
            )}
          </div>
        </Card>

        <Card
          title="Как выглядят галактики"
          description="Каждая точка — снимок галактики в координатах, сжимающих всю физику в две оси."
        >
          <div className="flex flex-col gap-4">
            <div className="flex gap-2">
              {splitKeys.map((key) => (
                <Button key={key} variant={activeScatter === key ? 'default' : 'outline'} onClick={() => setActiveScatter(key)}>
                  {splitLabels[key]}
                </Button>
              ))}
            </div>
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" dataKey="pcaX" name="Координата X" axisLine={false} tickLine={false} />
                  <YAxis type="number" dataKey="pcaY" name="Координата Y" axisLine={false} tickLine={false} />
                  <Tooltip
                    cursor={{ strokeDasharray: '3 3' }}
                    formatter={(value: number, name: string) => [value.toFixed(2), name]}
                    labelFormatter={() => ''}
                  />
                  {patternOrder.map((pattern) => (
                    <Scatter
                      key={pattern}
                      name={patternLabels[pattern]}
                      data={datasetForScatter.filter((sample) => sample.pattern === pattern)}
                      fill={getPatternColor(pattern)}
                      shape="circle"
                    />
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Card>
      </section>

      <section className="space-y-6">
        <Card
          title="Сконструируй архитектуру"
          description="Попробуй разные сочетания слоёв. Шумные данные требуют чуть более глубокой сети."
        >
          <div className="flex flex-col gap-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="rounded-lg border border-dashed border-primary/60 px-4 py-2 text-sm font-semibold text-primary">
                Вход: 4 признака (орбита, скорость, светимость, турбулентность)
              </div>
              <Button onClick={handleAddLayer}>Добавить слой</Button>
            </div>

            <DndContext sensors={layerSensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
              <SortableContext items={layers.map((layer) => layer.id)} strategy={verticalListSortingStrategy}>
                <div className="grid gap-4">
                  {layers.map((layer, index) => (
                    <SortableLayerCard
                      key={layer.id}
                      layer={layer}
                      index={index + 1}
                      onChange={handleLayerChange}
                      onRemove={handleLayerRemove}
                    />
                  ))}
                </div>
              </SortableContext>
            </DndContext>

            <NetworkOutputCard />
          </div>
        </Card>
        <ModelHints layers={layers} />
      </section>

      <section className="space-y-6">
        <Card title="Запусти обучение" description="Следи за процессом — хорошая модель стабильно держит высокую точность.">
          <div className="flex flex-col gap-6">
            <div className="max-w-sm space-y-2">
              <Label htmlFor="epochs">Сколько эпох обучать</Label>
              <div className="flex items-center gap-3">
                <Slider
                  id="epochs"
                  min={minEpochs}
                  max={maxEpochs}
                  step={1}
                  value={[epochCount]}
                  onValueChange={([value]) => setEpochCount(Math.round(value))}
                />
                <span className="text-sm font-semibold text-muted-foreground">{epochCount}</span>
              </div>
              <p className="text-xs text-muted-foreground">Максимум {maxEpochs} эпох — не даём сети переобучиться.</p>
            </div>
            <Button onClick={handleTrain} disabled={!canTrain} className="self-start">
              {isTraining ? 'Учимся…' : 'Стартовать обучение 🚀'}
            </Button>
            {summaryCards}
          </div>
        </Card>

        {decisionGrid && dataSplit && (
          <DecisionBoundaryCanvas
            grid={decisionGrid}
            points={dataSplit.test}
            getPointClass={(sample) => sample.pattern}
            getClassColor={getPatternColor}
            title="Как сеть чертит космические границы 🌌"
            description="Цветной фон показывает решение сети, точки — настоящие галактики из тестовой выборки."
          />
        )}

        {summary.testAccuracy !== null && (
          <Card title="Контрольная проверка" description="Смотрим, насколько хорошо сеть запоминает космические паттерны.">
            <div className="flex flex-col gap-6">
              <div className="rounded-2xl bg-gradient-to-r from-sky-200/50 via-indigo-200/40 to-purple-200/50 p-6 text-center">
                <p className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Точность на тесте</p>
                <p className="text-4xl font-black text-foreground">{formatPercent(summary.testAccuracy)}</p>
              </div>

              {summary.confusionMatrix && (
                <ConfusionMatrixTable
                  matrix={summary.confusionMatrix}
                  classOrder={patternOrder}
                  getClassLabel={(value) => patternLabels[value]}
                  getClassColor={getPatternColor}
                />
              )}

              {summary.samplePredictions.length > 0 && (
                <AdvancedPredictions predictions={summary.samplePredictions} />
              )}
            </div>
          </Card>
        )}
      </section>
    </div>
  );
}

interface GalaxySample {
  id: number;
  orbitRadius: number;
  orbitalSpeed: number;
  luminosity: number;
  turbulence: number;
  pcaX: number;
  pcaY: number;
  pattern: GalaxyClass;
}

type GalaxyClass = 'Orion' | 'Andromeda' | 'Centaurus';

type GalaxySplitData = Record<SplitKey, GalaxySample[]>;

interface GalaxyConfusionCell {
  actual: GalaxyClass;
  predicted: GalaxyClass;
  value: number;
}

interface AdvancedPrediction {
  sample: GalaxySample;
  predicted: GalaxyClass;
  correct: boolean;
}

interface AdvancedTrainingSummary {
  history: HistoryPoint[];
  accuracyHistory: HistoryPoint[];
  valHistory: HistoryPoint[];
  valAccuracyHistory: HistoryPoint[];
  testAccuracy: number | null;
  confusionMatrix: GalaxyConfusionCell[][] | null;
  samplePredictions: AdvancedPrediction[];
  featureStats: FeatureStats | null;
}

function toFeatureRow(sample: GalaxySample) {
  return [sample.orbitRadius, sample.orbitalSpeed, sample.luminosity, sample.turbulence];
}

function getPatternColor(pattern: GalaxyClass) {
  switch (pattern) {
    case 'Orion':
      return '#f97316';
    case 'Andromeda':
      return '#38bdf8';
    case 'Centaurus':
      return '#a855f7';
    default:
      return '#94a3b8';
  }
}

const patternLabels: Record<GalaxyClass, string> = {
  Orion: 'Спираль «Орион»',
  Andromeda: 'Туманность Андромеды',
  Centaurus: 'Рой Центавра',
};

interface AdvancedPredictionsProps {
  predictions: AdvancedPrediction[];
}

function AdvancedPredictions({ predictions }: AdvancedPredictionsProps) {
  return (
    <div className="overflow-hidden rounded-2xl border border-border shadow-sm">
      <table className="min-w-full divide-y divide-border text-sm">
        <thead className="bg-muted/60">
          <tr>
            <th className="px-4 py-3 text-left font-semibold text-muted-foreground">Орбитальный радиус</th>
            <th className="px-4 py-3 text-left font-semibold text-muted-foreground">Скорость вращения</th>
            <th className="px-4 py-3 text-left font-semibold text-muted-foreground">Светимость</th>
            <th className="px-4 py-3 text-left font-semibold text-muted-foreground">Турбулентность</th>
            <th className="px-4 py-3 text-left font-semibold text-muted-foreground">Настоящий класс</th>
            <th className="px-4 py-3 text-left font-semibold text-muted-foreground">Предсказание</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border bg-white/70">
          {predictions.map(({ sample, predicted, correct }, index) => (
            <tr key={`${sample.id}-${index}`} className={!correct ? 'bg-red-50/50' : undefined}>
              <td className="px-4 py-3 font-medium text-foreground">{sample.orbitRadius}</td>
              <td className="px-4 py-3 font-medium text-foreground">{sample.orbitalSpeed}</td>
              <td className="px-4 py-3 font-medium text-foreground">{sample.luminosity}</td>
              <td className="px-4 py-3 font-medium text-foreground">{sample.turbulence}</td>
              <td className="px-4 py-3 font-medium text-foreground">
                <span className="flex items-center gap-2">
                  <span className="h-3 w-3 rounded-full" style={{ background: getPatternColor(sample.pattern) }} />
                  {patternLabels[sample.pattern]}
                </span>
              </td>
              <td className="px-4 py-3 font-medium text-foreground">
                <span className="flex items-center gap-2">
                  <span className="h-3 w-3 rounded-full" style={{ background: getPatternColor(predicted) }} />
                  {patternLabels[predicted]} {correct ? '✅' : '❌'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function generateGalaxyDecisionGrid(
  model: tf.LayersModel,
  means: number[],
  std: number[],
  samples: GalaxySample[]
): DecisionGridData<GalaxyClass> {
  if (!samples.length) {
    return { grid: [], minX: 0, maxX: 0, minY: 0, maxY: 0, classOrder: patternOrder };
  }

  const gridSize = 60;
  const xs = samples.map((sample) => sample.pcaX);
  const ys = samples.map((sample) => sample.pcaY);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const decision: number[][] = [];

  for (let xi = 0; xi < gridSize; xi++) {
    const row: number[] = [];
    for (let yi = 0; yi < gridSize; yi++) {
      const pcaX = minX + (xi / (gridSize - 1)) * (maxX - minX);
      const pcaY = minY + (yi / (gridSize - 1)) * (maxY - minY);
      const approxFeatures = estimateFeaturesFromNeighbors(pcaX, pcaY, samples);
      const standardized = approxFeatures.map((value, index) => (value - means[index]) / std[index]);
      const input = tf.tensor2d([standardized]);
      const prediction = model.predict(input) as tf.Tensor;
      const probabilities = prediction.dataSync();
      input.dispose();
      prediction.dispose();
      const predictedIndex = probabilities.indexOf(Math.max(...probabilities));
      row.push(predictedIndex);
    }
    decision.push(row);
  }

  return { grid: decision, minX, maxX, minY, maxY, classOrder: patternOrder };
}

function estimateFeaturesFromNeighbors(pcaX: number, pcaY: number, samples: GalaxySample[]) {
  const neighborCount = Math.min(6, samples.length);
  const neighbors = samples
    .map((sample) => {
      const dx = sample.pcaX - pcaX;
      const dy = sample.pcaY - pcaY;
      const distance = Math.sqrt(dx * dx + dy * dy);
      return { sample, distance };
    })
    .sort((a, b) => a.distance - b.distance)
    .slice(0, neighborCount);

  if (!neighbors.length) {
    return [0, 0, 0, 0];
  }

  const features = [0, 0, 0, 0];
  let weightSum = 0;

  neighbors.forEach(({ sample, distance }) => {
    const weight = 1 / Math.max(distance, 0.05);
    weightSum += weight;
    features[0] += sample.orbitRadius * weight;
    features[1] += sample.orbitalSpeed * weight;
    features[2] += sample.luminosity * weight;
    features[3] += sample.turbulence * weight;
  });

  if (weightSum === 0) {
    const fallback = neighbors[0]?.sample ?? samples[0];
    return [fallback.orbitRadius, fallback.orbitalSpeed, fallback.luminosity, fallback.turbulence];
  }

  return features.map((value) => value / weightSum);
}
