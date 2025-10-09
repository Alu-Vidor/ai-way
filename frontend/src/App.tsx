import { useMemo, useState } from 'react';
import { Button } from './components/ui/button';
import { IrisLab } from './views/IrisLab';
import { AdvancedLab } from './views/AdvancedLab';

type TabKey = 'iris' | 'advanced';

interface TabConfig {
  id: TabKey;
  label: string;
  description: string;
}

const tabs: TabConfig[] = [
  {
    id: 'iris',
    label: 'Разминка на ирисах',
    description: 'Лёгкий старт: три вида цветов, понятные признаки и наглядные границы.',
  },
  {
    id: 'advanced',
    label: 'Продвинутая миссия',
    description: 'Брось вызов: шумные галактики с пересекающимися классами и сложными зависимостями.',
  },
];

function App() {
  const [activeTab, setActiveTab] = useState<TabKey>('iris');

  const activeDescription = useMemo(() => tabs.find((tab) => tab.id === activeTab)?.description ?? '', [activeTab]);

  return (
    <div className="min-h-screen bg-slate-100">
      <div className="bg-gradient-to-r from-indigo-500 via-sky-500 to-cyan-400 text-white">
        <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-12 text-center md:text-left">
          <div className="space-y-3">
            <p className="text-sm font-semibold uppercase tracking-widest text-white/80">Мастерская нейросетей</p>
            <h1 className="text-3xl font-black md:text-4xl">
              Собери архитектуру и доведи точность до максимума
            </h1>
            <p className="max-w-3xl text-base text-white/90 md:text-lg">
              Выбирай площадку: начни с классики про цветы или переходи сразу к космическому челленджу. Все параметры —
              в твоих руках.
            </p>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-3 md:justify-start">
            {tabs.map((tab) => {
              const isActive = tab.id === activeTab;
              return (
                <Button
                  key={tab.id}
                  variant={isActive ? 'default' : 'outline'}
                  className={
                    isActive
                      ? 'bg-white/90 text-slate-900 hover:bg-white'
                      : 'border-white/50 text-white hover:bg-white/10 hover:text-white'
                  }
                  onClick={() => setActiveTab(tab.id)}
                >
                  {tab.label}
                </Button>
              );
            })}
          </div>

          <p className="text-sm text-white/80 md:text-base">{activeDescription}</p>
        </div>
      </div>

      <main className="pb-16">
        {activeTab === 'iris' ? <IrisLab /> : <AdvancedLab />}
      </main>
    </div>
  );
}

export default App;
