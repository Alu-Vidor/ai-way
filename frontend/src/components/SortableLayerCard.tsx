import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { GripVertical, Trash2 } from 'lucide-react';
import { LayerConfig } from '../types';
import { Button } from './ui/button';
import { Label } from './ui/label';
import { Slider } from './ui/slider';

interface SortableLayerCardProps {
  layer: LayerConfig;
  index: number;
  onChange: (id: string, changes: Partial<LayerConfig>) => void;
  onRemove: (id: string) => void;
}

const activations: LayerConfig['activation'][] = ['relu', 'sigmoid', 'softmax'];

export function SortableLayerCard({ layer, index, onChange, onRemove }: SortableLayerCardProps) {
  const { attributes, listeners, setNodeRef, transform, transition } = useSortable({ id: layer.id });

  return (
    <div
      ref={setNodeRef}
      style={{ transform: CSS.Transform.toString(transform), transition }}
      className="card border-2 border-dashed border-primary/30 bg-white/80 p-4 shadow-sm"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-primary">
          <button
            {...attributes}
            {...listeners}
            className="flex h-8 w-8 items-center justify-center rounded-full border border-primary/30 bg-white text-primary shadow-sm"
            aria-label="Переместить слой"
          >
            <GripVertical className="h-4 w-4" />
          </button>
          <span>Слой {index}</span>
        </div>
        <Button variant="ghost" onClick={() => onRemove(layer.id)} className="text-destructive" aria-label="Удалить слой">
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>

      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <Label htmlFor={`${layer.id}-units`}>Нейронов</Label>
          <input
            id={`${layer.id}-units`}
            type="number"
            min={1}
            max={128}
            value={layer.units}
            onChange={(event) => onChange(layer.id, { units: Number(event.target.value) })}
            className="w-full rounded-lg border border-border px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor={`${layer.id}-activation`}>Активация</Label>
          <select
            id={`${layer.id}-activation`}
            value={layer.activation}
            onChange={(event) => onChange(layer.id, { activation: event.target.value as LayerConfig['activation'] })}
            className="w-full rounded-lg border border-border px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          >
            {activations.map((activation) => (
              <option key={activation} value={activation}>
                {activation.toUpperCase()}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <Label>Отсев (dropout)</Label>
          <div className="flex items-center gap-3">
            <Slider
              min={0}
              max={0.7}
              step={0.05}
              value={[layer.dropout]}
              onValueChange={([value]) => onChange(layer.id, { dropout: Number(value) })}
            />
            <span className="text-sm font-semibold text-muted-foreground">{Math.round(layer.dropout * 100)}%</span>
          </div>
        </div>

        <div className="space-y-2">
          <Label>Нормализация батча</Label>
          <label className="flex items-center gap-3 text-sm font-medium text-muted-foreground">
            <input
              type="checkbox"
              checked={layer.batchNorm}
              onChange={(event) => onChange(layer.id, { batchNorm: event.target.checked })}
              className="h-4 w-4 rounded border-border text-primary focus:ring-primary"
            />
            Выровнять распределение признаков
          </label>
        </div>
      </div>
    </div>
  );
}
