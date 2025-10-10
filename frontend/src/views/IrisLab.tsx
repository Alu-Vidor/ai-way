import { useCallback, useMemo, useState } from 'react';
import { DndContext, closestCenter, KeyboardSensor, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import type { DragEndEvent } from '@dnd-kit/core';
import { arrayMove, SortableContext, sortableKeyboardCoordinates, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { v4 as uuid } from 'uuid';
import * as tf from '@tensorflow/tfjs';
import irisData from '../assets/iris.json';
import pcaParams from '../assets/pca-params.json';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import { Label } from '../components/ui/label';
import { Slider } from '../components/ui/slider';
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
import { buildPieData, formatPercent, getClassColor, seedShuffle } from '../lib/utils';
import {
  Species,
  IrisSample,
  LayerConfig,
  SplitData,
  HistoryPoint,
  ConfusionMatrixCell,
  TrainingSummary,
  DecisionGridData,
  splitLabels,
  SplitKey,
  speciesLabels,
} from '../types';
import { DecisionBoundaryCanvas } from '../components/DecisionBoundaryCanvas';
import { ConfusionMatrixTable } from '../components/ConfusionMatrixTable';
import { SamplePredictions } from '../components/SamplePredictions';
import { ModelHints } from '../components/ModelHints';

const speciesOrder: Species[] = ['Setosa', 'Versicolor', 'Virginica'];
const splitKeys: SplitKey[] = ['train', 'val', 'test'];

const defaultLayers: LayerConfig[] = [
  {
    id: uuid(),
    units: 8,
    activation: 'relu',
    dropout: 0,
    batchNorm: false,
  },
];

const minSplit = 10;
const epochs = 30;

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
  const [splits, setSplits] = useState({ train: 60, val: 20, test: 20 });

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

const initialState: TrainingSummary = {
  history: [],
  accuracyHistory: [],
  valHistory: [],
  valAccuracyHistory: [],
  testAccuracy: null,
  confusionMatrix: null,
  samplePredictions: [],
  featureStats: null,
};

export function IrisLab() {
  const layerSensors = useLayerSensors();
  const { splits, updateTrain, updateVal } = useSplitState();
  const [seed, setSeed] = useState(42);
  const [dataReady, setDataReady] = useState(false);
  const [dataSplit, setDataSplit] = useState<SplitData | null>(null);
  const [layers, setLayers] = useState<LayerConfig[]>(defaultLayers);
  const [isTraining, setIsTraining] = useState(false);
  const [summary, setSummary] = useState<TrainingSummary>(initialState);
  const [activeScatter, setActiveScatter] = useState<SplitKey>('train');
  const [decisionGrid, setDecisionGrid] = useState<DecisionGridData<Species> | null>(null);

  const samples = useMemo(() => {
    return (irisData as IrisSample[]).map((item, index) => ({ ...item, id: index }));
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
        units: 6,
        activation: 'relu',
        dropout: 0,
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
    setDecisionGrid(null);
    setSummary(initialState);

    const disposeTensors: tf.Tensor[] = [];

    try {
      const labelToIndex: Record<Species, number> = {
        Setosa: 0,
        Versicolor: 1,
        Virginica: 2,
      };

      const trainFeatures = dataSplit.train.map((sample) => [
        sample.sepalLength,
        sample.sepalWidth,
        sample.petalLength,
        sample.petalWidth,
      ]);
      const valFeatures = dataSplit.val.map((sample) => [
        sample.sepalLength,
        sample.sepalWidth,
        sample.petalLength,
        sample.petalWidth,
      ]);
      const testFeatures = dataSplit.test.map((sample) => [
        sample.sepalLength,
        sample.sepalWidth,
        sample.petalLength,
        sample.petalWidth,
      ]);

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

      const encodeLabels = (samples: IrisSample[]) =>
        samples.map((sample) => {
          const vector = [0, 0, 0];
          vector[labelToIndex[sample.species]] = 1;
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

      model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

      model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
      });

      const historyPoints: HistoryPoint[] = [];
      const accuracyPoints: HistoryPoint[] = [];
      const valHistoryPoints: HistoryPoint[] = [];
      const valAccuracyPoints: HistoryPoint[] = [];

      await model.fit(trainXs, trainYs, {
        epochs,
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

      let confusionMatrix: ConfusionMatrixCell[][] | null = null;
      let samplesPreview: TrainingSummary['samplePredictions'] = [];

      if (testXs && testYs && dataSplit.test.length) {
        const predictions = model.predict(testXs) as tf.Tensor;
        disposeTensors.push(predictions);
        const predArray = predictions.arraySync() as number[][];
        const predictedIndices = predArray.map((row) => row.indexOf(Math.max(...row)));
        const confusion = Array.from({ length: 3 }, () => Array(3).fill(0));
        dataSplit.test.forEach((sample, index) => {
          const predictedIndex = predictedIndices[index];
          const actualIndex = labelToIndex[sample.species];
          confusion[actualIndex][predictedIndex] += 1;
        });
        confusionMatrix = confusion.map((row, actual) =>
          row.map((value, predicted) => ({
            actual: speciesOrder[actual],
            predicted: speciesOrder[predicted],
            value,
          }))
        );

        const previewSource = dataSplit.test.map((sample, index) => ({ sample, predictedIndex: predictedIndices[index] }));
        const rng = seedShuffle(previewSource, seed).slice(0, 5);
        samplesPreview = rng.map(({ sample, predictedIndex }) => ({
          sample,
          predicted: speciesOrder[predictedIndex],
          correct: speciesOrder[predictedIndex] === sample.species,
        }));
      }

      const grid = generateDecisionGrid(model, featureMeans, featureStd);

      setDecisionGrid(grid);
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
  }, [dataSplit, isTraining, layers, seed]);

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
        <Card title="Как меняется ошибка" description="Ошибка показывает, как сеть учится. Чем ниже — тем лучше.">
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="epoch" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} />
                <Tooltip formatter={(value: number) => value.toFixed(3)} labelFormatter={(label) => `Эпоха ${label}`} />
                <Legend />
                <Line type="monotone" dataKey="loss" name="Обучение" stroke="#6366f1" strokeWidth={2} dot={false} />
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

        <Card title="Как растёт точность" description="Точность — сколько цветов сеть угадывает правильно.">
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
                    stroke="#f97316"
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
        <h1 className="text-4xl font-black text-primary">Нейросеть учится различать цветы 🌸</h1>
        <p className="mx-auto max-w-3xl text-lg text-muted-foreground">
          Это набор данных про три вида ирисов — цветов. Каждый цветок описан числами: длина и ширина лепестков и чашелистиков.
          Слева — эти данные, справа — как нейросеть «разделяет» их.
        </p>
      </header>

      <section className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
        <Card title="Загрузи данные про цветы" description="Настрой пропорции и нажми кнопку, чтобы увидеть набор.">
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
                  max={80}
                  step={5}
                  onValueChange={([value]) => updateTrain(value)}
                />
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label htmlFor="val">{splitLabels.val}</Label>
                  <span className="text-sm font-semibold text-foreground">{splits.val}%</span>
                </div>
                <Slider id="val" value={[splits.val]} min={minSplit} max={45} step={5} onValueChange={([value]) => updateVal(value)} />
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

            <Button onClick={handleApplyData}>Загрузить данные про цветы</Button>

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
                {speciesOrder.map((species) => (
                  <span
                    key={species}
                    className="flex items-center gap-2 rounded-full bg-muted px-3 py-1 font-medium text-muted-foreground"
                  >
                    <span className="h-3 w-3 rounded-full" style={{ background: getClassColor(species) }} />
                    {speciesLabels[species]}
                  </span>
                ))}
              </div>
            )}
          </div>
        </Card>

        <Card title="Как выглядят цветы" description="Каждая точка — цветок. Цвет точки — вид ириса.">
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
                  <XAxis type="number" dataKey="pcaX" name="Главный признак 1" axisLine={false} tickLine={false} />
                  <YAxis type="number" dataKey="pcaY" name="Главный признак 2" axisLine={false} tickLine={false} />
                  <Tooltip
                    cursor={{ strokeDasharray: '3 3' }}
                    formatter={(value: number, name: string) => [value.toFixed(2), name]}
                    labelFormatter={() => ''}
                  />
                  {speciesOrder.map((species) => (
                    <Scatter
                      key={species}
                      name={speciesLabels[species]}
                      data={datasetForScatter.filter((sample) => sample.species === species)}
                      fill={getClassColor(species)}
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
          title="Собери свою нейросеть"
          description="Добавляй слои, настраивай нейроны и активации. Ошибки подсветятся красным."
        >
          <div className="flex flex-col gap-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="rounded-lg border border-dashed border-primary/60 px-4 py-2 text-sm font-semibold text-primary">
                Вход: 4 числа (признаки цветка)
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
        <Card title="Запусти обучение" description="Следи за прогрессом по эпохам и смотри, как сеть учится.">
          <div className="flex flex-col gap-6">
            <Button onClick={handleTrain} disabled={!canTrain} className="self-start">
              {isTraining ? 'Учимся…' : 'Обучить сеть 🚀'}
            </Button>
            {summaryCards}
          </div>
        </Card>

        {decisionGrid && dataSplit && summary.featureStats && (
          <DecisionBoundaryCanvas
            grid={decisionGrid}
            points={dataSplit.test}
            getPointClass={(point) => point.species}
            getClassColor={getClassColor}
            title="Как сеть рисует границы ✍️"
            description="Цветной фон — решение сети, точки — настоящие ирисы из теста. Если точка в чужой зоне, сеть ошиблась."
          />
        )}

        {summary.testAccuracy !== null && (
          <Card title="Проверка знаний" description="Смотрим, что сеть умеет на отложенных данных.">
            <div className="flex flex-col gap-6">
              <div className="rounded-2xl bg-gradient-to-r from-primary/10 via-secondary/10 to-accent/10 p-6 text-center">
                <p className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Точность на тесте</p>
                <p className="text-4xl font-black text-foreground">{formatPercent(summary.testAccuracy)}</p>
              </div>

              {summary.confusionMatrix && (
                <ConfusionMatrixTable
                  matrix={summary.confusionMatrix}
                  classOrder={speciesOrder}
                  getClassLabel={(value) => speciesLabels[value]}
                  getClassColor={getClassColor}
                />
              )}

              {summary.samplePredictions.length > 0 && (
                <SamplePredictions predictions={summary.samplePredictions} />
              )}
            </div>
          </Card>
        )}
      </section>
    </div>
  );
}

function generateDecisionGrid(model: tf.LayersModel, means: number[], std: number[]): DecisionGridData<Species> {
  const gridSize = 60;
  const xs = (irisData as IrisSample[]).map((sample) => sample.pcaX);
  const ys = (irisData as IrisSample[]).map((sample) => sample.pcaY);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const componentMatrix = pcaParams.components as number[][];
  const pcaMean = pcaParams.pcaMean as number[];
  const originalMean = pcaParams.mean as number[];
  const originalScale = pcaParams.scale as number[];

  const decision: number[][] = [];

  for (let xi = 0; xi < gridSize; xi++) {
    const row: number[] = [];
    for (let yi = 0; yi < gridSize; yi++) {
      const pcaX = minX + (xi / (gridSize - 1)) * (maxX - minX);
      const pcaY = minY + (yi / (gridSize - 1)) * (maxY - minY);
      const scaled = componentMatrix[0].map((_, featureIndex) => {
        const value =
          pcaMean[featureIndex] + pcaX * componentMatrix[0][featureIndex] + pcaY * componentMatrix[1][featureIndex];
        return value;
      });
      const original = scaled.map((value, featureIndex) => value * originalScale[featureIndex] + originalMean[featureIndex]);
      const standardized = original.map((value, featureIndex) => (value - means[featureIndex]) / std[featureIndex]);
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

  return { grid: decision, minX, maxX, minY, maxY, classOrder: speciesOrder };
}
