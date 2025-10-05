import { LayerConfig } from '../types';

interface ModelHintsProps {
  layers: LayerConfig[];
}

export function ModelHints({ layers }: ModelHintsProps) {
  const issues: string[] = [];

  if (layers.length === 0) {
    issues.push('Добавь хотя бы один скрытый слой, чтобы сеть могла учиться.');
  }

  const softmaxInside = layers.slice(0, -1).some((layer) => layer.activation === 'softmax');
  if (softmaxInside) {
    issues.push('Softmax лучше оставить только в конце — иначе границы получаются странными.');
  }

  const heavyDropout = layers.some((layer) => layer.dropout > 0.5);
  if (heavyDropout) {
    issues.push('Слишком большой dropout может «забывать» важные признаки. Попробуй 0.2–0.3.');
  }

  if (issues.length === 0) {
    return (
      <div className="rounded-xl border border-border bg-emerald-50/60 p-4 text-sm text-emerald-700">
        Всё отлично! Сеть готова к обучению.
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-destructive/40 bg-red-50/60 p-4 text-sm text-destructive">
      <p className="mb-2 font-semibold">Подсказки:</p>
      <ul className="list-disc space-y-1 pl-5">
        {issues.map((issue, index) => (
          <li key={index}>{issue}</li>
        ))}
      </ul>
    </div>
  );
}
