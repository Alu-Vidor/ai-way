import { useMemo } from 'react';
import { DecisionGridData, IrisSample, Species } from '../types';
import { getClassColor } from '../lib/utils';

interface DecisionBoundaryCanvasProps {
  grid: DecisionGridData;
  points: IrisSample[];
}

const speciesOrder: Species[] = ['Setosa', 'Versicolor', 'Virginica'];

export function DecisionBoundaryCanvas({ grid, points }: DecisionBoundaryCanvasProps) {
  const canvasData = useMemo(() => {
    if (!grid.grid.length || !grid.grid[0]?.length) {
      return null;
    }
    const width = 600;
    const height = 400;
    const cells: Array<{ x: number; y: number; color: string }> = [];
    const { grid: matrix, minX, maxX, minY, maxY } = grid;
    const stepX = width / matrix.length;
    const stepY = height / matrix[0].length;
    matrix.forEach((row, ix) => {
      row.forEach((value, iy) => {
        cells.push({
          x: ix * stepX,
          y: height - (iy + 1) * stepY,
          color: getClassColor(speciesOrder[value]),
        });
      });
    });
    const scaledPoints = points.map((point) => ({
      x: ((point.pcaX - minX) / (maxX - minX)) * width,
      y: height - ((point.pcaY - minY) / (maxY - minY)) * height,
      species: point.species,
    }));
    return { width, height, cells, scaledPoints, stepX, stepY };
  }, [grid, points]);

  if (!canvasData) {
    return null;
  }

  return (
    <div className="card flex flex-col gap-4">
      <div className="space-y-2">
        <h3 className="text-2xl font-semibold text-foreground">Как сеть рисует границы ✍️</h3>
        <p className="text-sm text-muted-foreground">
          Цветной фон — решение сети, точки — настоящие цветы из теста. Если точка в «чужой» зоне, сеть ошиблась.
        </p>
      </div>
      <div className="overflow-hidden rounded-2xl border border-border bg-white shadow">
        <svg viewBox={`0 0 ${canvasData.width} ${canvasData.height}`} className="w-full">
          {canvasData.cells.map((cell, index) => (
            <rect key={index} x={cell.x} y={cell.y} width={canvasData.stepX} height={canvasData.stepY} fill={cell.color} opacity={0.5} />
          ))}
          {canvasData.scaledPoints.map((point, index) => (
            <circle key={index} cx={point.x} cy={point.y} r={6} fill={getClassColor(point.species)} stroke="#0f172a" strokeWidth={1.5} />
          ))}
        </svg>
      </div>
    </div>
  );
}
