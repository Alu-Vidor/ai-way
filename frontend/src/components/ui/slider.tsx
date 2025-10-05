import * as SliderPrimitive from '@radix-ui/react-slider';
import { clsx } from 'clsx';

export type SliderProps = SliderPrimitive.SliderProps;

export function Slider({ className, ...props }: SliderProps) {
  return (
    <SliderPrimitive.Root
      className={clsx('relative flex w-full touch-none select-none items-center', className)}
      {...props}
    >
      <SliderPrimitive.Track className="relative h-2 w-full grow overflow-hidden rounded-full bg-muted">
        <SliderPrimitive.Range className="absolute h-full bg-primary" />
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb className="block h-5 w-5 rounded-full border-2 border-primary bg-white shadow transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary" />
    </SliderPrimitive.Root>
  );
}
