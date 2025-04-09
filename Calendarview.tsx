
import { format, parseISO } from 'date-fns';

interface PriceEntry {
  dep: string;
  ret: string;
  price: number;
}

interface Props {
  prices: PriceEntry[];
}

export default function CalendarView({ prices }: Props) {
  if (!prices?.length) return null;

  const bestPrice = Math.min(...prices.map(p => p.price));

  // Extract unique sorted dates
  const depDates = [...new Set(prices.map(p => p.dep))].sort();
  const retDates = [...new Set(prices.map(p => p.ret))].sort();

  const getPrice = (dep: string, ret: string) =>
    prices.find(p => p.dep === dep && p.ret === ret)?.price;

  return (
    <div className="mt-10 overflow-auto max-w-full">
      <h3 className="text-xl font-bold text-white mb-4 text-center">📆 Flexible Dates Matrix</h3>
      <table className="min-w-[500px] text-sm text-white border-collapse border border-white/20">
        <thead>
          <tr>
            <th className="bg-white/10 p-2 text-left">Depart ↓ / Return →</th>
            {retDates.map((ret) => (
              <th key={ret} className="p-2 bg-white/10 text-center">
                {format(parseISO(ret), 'MMM dd')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {depDates.map((dep) => (
            <tr key={dep}>
              <td className="p-2 bg-white/10 font-medium">{format(parseISO(dep), 'MMM dd')}</td>
              {retDates.map((ret) => {
                const price = getPrice(dep, ret);
                const isBest = price === bestPrice;
                return (
                  <td
                    key={`${dep}-${ret}`}
                    className={`p-2 text-center border border-white/10 ${
                      price
                        ? isBest
                          ? 'bg-green-600 font-bold text-white'
                          : 'bg-blue-700/40 hover:bg-blue-600'
                        : 'text-white/30 bg-white/5'
                    }`}
                  >
                    {price ? `$${price}` : '—'}
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
