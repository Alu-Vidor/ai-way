import { clsx } from 'clsx';
import { type HTMLAttributes } from 'react';

type CardProps = HTMLAttributes<HTMLDivElement> & {
  title?: string;
  description?: string;
};

export function Card({ className, title, description, children, ...props }: CardProps) {
  return (
    <div className={clsx('card flex flex-col gap-4 p-6', className)} {...props}>
      {(title || description) && (
        <div className="space-y-1">
          {title && <h3 className="text-lg font-semibold text-foreground">{title}</h3>}
          {description && <p className="text-sm text-muted-foreground">{description}</p>}
        </div>
      )}
      {children}
    </div>
  );
}
