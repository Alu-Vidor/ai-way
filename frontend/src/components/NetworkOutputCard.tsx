export function NetworkOutputCard() {
  return (
    <div className="card border-2 border-primary/40 bg-gradient-to-r from-primary/10 to-secondary/10">
      <div className="flex flex-col gap-2">
        <h4 className="text-lg font-semibold text-primary">Выходной слой</h4>
        <p className="text-sm text-muted-foreground">
          3 нейрона и активация <span className="font-semibold text-primary">Softmax</span> — по одному на каждый вид цветка.
        </p>
      </div>
    </div>
  );
}
