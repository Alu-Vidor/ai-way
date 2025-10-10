import { useMemo } from 'react';

interface ConfusionMatrixTableProps<TClass extends string> {
  matrix: Array<Array<{ actual: TClass; predicted: TClass; value: number }>>;
  classOrder: TClass[];
  getClassLabel: (value: TClass) => string;
  getClassColor: (value: TClass) => string;
}

export function ConfusionMatrixTable<TClass extends string>({
  matrix,
  classOrder,
  getClassLabel,
  getClassColor,
}: ConfusionMatrixTableProps<TClass>) {
  const { maxValue, totals, rowMap } = useMemo(() => {
    let max = 0;
    const totalPerActual = {} as Record<TClass, number>;
    const matrixMap = {} as Record<TClass, Record<TClass, number>>;

    classOrder.forEach((actual) => {
      totalPerActual[actual] = 0;
      matrixMap[actual] = {} as Record<TClass, number>;
      classOrder.forEach((predicted) => {
        matrixMap[actual][predicted] = 0;
      });
    });

    matrix.forEach((row) => {
      row.forEach((cell) => {
        if (cell.value > max) {
          max = cell.value;
        }
        totalPerActual[cell.actual] += cell.value;
        matrixMap[cell.actual][cell.predicted] = cell.value;
      });
    });

    return { maxValue: Math.max(max, 1), totals: totalPerActual, rowMap: matrixMap };
  }, [classOrder, matrix]);

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-separate border-spacing-0 text-sm">
        <thead>
          <tr>
            <th className="bg-white px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Настоящий \ Предсказанный
            </th>
            {classOrder.map((cls) => (
              <th
                key={cls}
                className="bg-muted/40 px-4 py-3 text-center text-xs font-semibold uppercase tracking-wide text-muted-foreground"
              >
                {getClassLabel(cls)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {classOrder.map((actual) => (
            <tr key={actual}>
              <th className="bg-muted/30 px-4 py-3 text-left font-semibold text-muted-foreground">
                {getClassLabel(actual)}
                <span className="ml-2 text-xs font-normal text-muted-foreground/70">
                  {`${totals[actual] ?? 0} шт.`}
                </span>
              </th>
              {classOrder.map((predicted) => {
                const value = rowMap[actual]?.[predicted] ?? 0;
                const intensity = value / maxValue;
                const background = mixColor(getClassColor(predicted), 0.18 + intensity * 0.55);
                const textColor = intensity > 0.55 ? '#0f172a' : '#0f172a';
                return (
                  <td key={predicted} className="px-4 py-3 text-center" style={{ backgroundColor: background, color: textColor }}>
                    <div className="flex flex-col items-center gap-1">
                      <span className="text-base font-semibold">{value}</span>
                      <span className="text-[11px] uppercase tracking-wide text-muted-foreground/70">
                        {getClassLabel(predicted)}
                      </span>
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function mixColor(hex: string, alpha: number) {
  const { r, g, b } = hexToRgb(hex);
  const clampedAlpha = Math.min(Math.max(alpha, 0.1), 0.85);
  return `rgba(${r}, ${g}, ${b}, ${clampedAlpha})`;
}

function hexToRgb(hex: string) {
  let value = hex.replace('#', '');
  if (value.length === 3) {
    value = value
      .split('')
      .map((char) => char + char)
      .join('');
  }
  const numeric = Number.parseInt(value, 16);
  const r = (numeric >> 16) & 255;
  const g = (numeric >> 8) & 255;
  const b = numeric & 255;
  return { r, g, b };
}
