import { SamplePredictionRow } from '../types';
import { getClassColor } from '../lib/utils';

interface SamplePredictionsProps {
  predictions: SamplePredictionRow[];
}

export function SamplePredictions({ predictions }: SamplePredictionsProps) {
  return (
    <div className="overflow-hidden rounded-2xl border border-border shadow-sm">
      <table className="min-w-full divide-y divide-border text-sm">
        <thead className="bg-muted/60">
          <tr>
            <th className="px-4 py-3 text-left font-semibold text-muted-foreground">SepalLength</th>
            <th className="px-4 py-3 text-left font-semibold text-muted-foreground">SepalWidth</th>
            <th className="px-4 py-3 text-left font-semibold text-muted-foreground">Настоящий вид</th>
            <th className="px-4 py-3 text-left font-semibold text-muted-foreground">Предсказано</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border bg-white/70">
          {predictions.map(({ sample, predicted, correct }, index) => (
            <tr key={`${sample.id}-${index}`} className={!correct ? 'bg-red-50/50' : undefined}>
              <td className="px-4 py-3 font-medium text-foreground">{sample.sepalLength}</td>
              <td className="px-4 py-3 font-medium text-foreground">{sample.sepalWidth}</td>
              <td className="px-4 py-3 font-medium text-foreground">
                <span className="flex items-center gap-2">
                  <span className="h-3 w-3 rounded-full" style={{ background: getClassColor(sample.species) }} />
                  {sample.species}
                </span>
              </td>
              <td className="px-4 py-3 font-medium text-foreground">
                <span className="flex items-center gap-2">
                  <span className="h-3 w-3 rounded-full" style={{ background: getClassColor(predicted) }} />
                  {predicted} {correct ? '✅' : '❌'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
