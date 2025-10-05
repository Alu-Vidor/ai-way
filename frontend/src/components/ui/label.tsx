import { forwardRef } from 'react';
import { Label as RadixLabel, type LabelProps as RadixLabelProps } from '@radix-ui/react-label';
import { clsx } from 'clsx';

type LabelProps = RadixLabelProps;

export const Label = forwardRef<HTMLLabelElement, LabelProps>(
  ({ className, ...props }, ref) => (
    <RadixLabel ref={ref} className={clsx('text-sm font-medium text-muted-foreground', className)} {...props} />
  )
);

Label.displayName = 'Label';
