import { useState } from 'react';
import axios from 'axios';
import { format } from 'date-fns';
import { Line } from 'react-chartjs-2';
import CalendarView from './CalendarView';
import RouteMap from './Map';

import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Tooltip, Legend);

interface PredictionResult {
  best_booking_date: string;
  predicted_price: number;
  kayak_price: string;
  booking_advice: string;
  kayak_link: string;
  price_by_day: [number, number][];
  price_by_duration: [number, number][];
  calendar_prices: { dep: string; ret: string; price: number }[];
}

const airports = [
  { code: 'YYZ', name: 'Toronto' },
  { code: 'YVR', name: 'Vancouver' },
  { code: 'YUL', name: 'Montreal' },
  { code: 'YYC', name: 'Calgary' },
  { code: 'YEG', name: 'Edmonton' },
  { code: 'BCN', name: 'Barcelona' },
  { code: 'FCO', name: 'Rome' },
  { code: 'ZRH', name: 'Zurich' },
  { code: 'LIS', name: 'Lisbon' },
  { code: 'BRU', name: 'Brussels' },
];

export default function App() {
  const [origin, setOrigin] = useState('YUL');
  const [destination, setDestination] = useState('BCN');
  const [departure, setDeparture] = useState('2025-08-20');
  const [arrival, setArrival] = useState('2025-08-30');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [activeChart, setActiveChart] = useState<'lead' | 'duration'>('lead');

  const handlePredict = async () => {
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:5000/predict', {
        origin,
        destination,
        departure_date: departure,
        arrival_date: arrival,
      });
      setResult(res.data);
    } catch (err) {
      alert('Prediction failed. Ensure Flask is running.');
    }
    setLoading(false);
  };

  const leadChart = {
    labels: result?.price_by_day.map((d) => d[0]),
    datasets: [
      {
        label: 'Price vs Days Before Departure',
        data: result?.price_by_day.map((d) => d[1]),
        borderColor: '#3b82f6',
        backgroundColor: '#60a5fa',
        tension: 0.4,
      },
    ],
  };

  const durationChart = {
    labels: result?.price_by_duration.map((d) => d[0]),
    datasets: [
      {
        label: 'Price vs Trip Duration',
        data: result?.price_by_duration.map((d) => d[1]),
        borderColor: '#10b981',
        backgroundColor: '#6ee7b7',
        tension: 0.4,
      },
    ],
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-black text-white p-6 flex flex-col items-center">
      <div className="max-w-4xl w-full text-center">
        
        <h1 className="text-4xl font-bold mb-6">📅 When to Book? We've Got the Answer.</h1>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6 text-black">
          <select value={origin} onChange={(e) => setOrigin(e.target.value)} className="p-3 rounded">
            {airports.map((a) => (
              <option key={a.code} value={a.code}>
                {a.name.toUpperCase()} ({a.code})
              </option>
            ))}
          </select>
          <select value={destination} onChange={(e) => setDestination(e.target.value)} className="p-3 rounded">
            {airports.map((a) => (
              <option key={a.code} value={a.code}>
                {a.name.toUpperCase()} ({a.code})
              </option>
            ))}
          </select>
          <input type="date" value={departure} onChange={(e) => setDeparture(e.target.value)} className="p-3 rounded" />
          <input type="date" value={arrival} onChange={(e) => setArrival(e.target.value)} className="p-3 rounded" />
        </div>

        <button
          onClick={handlePredict}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-3 rounded-full shadow transition"
        >
          {loading ? '🔄 Predicting...' : '📊 Get Price Forecast'}
        </button>

        {loading && <div className="mt-4 animate-pulse text-blue-300">Loading, please wait...</div>}

        {result && (
          <>
            <div className="mt-10 bg-white/10 p-6 rounded-xl text-left space-y-2">
              <h2 className="text-xl font-bold">Prediction Summary</h2>
              <p>📅 Best Time to Book: <strong>{format(new Date(result.best_booking_date), 'MMM dd')}</strong></p>
              <p>📉 Predicted Price: <strong>${result.predicted_price}</strong></p>
            
             
            </div>

            <div className="mt-10">
              <CalendarView prices={result.calendar_prices} />
            </div>

            <div className="mt-10 flex justify-center space-x-4">
              <button
                className={`px-4 py-2 rounded-full text-sm ${
                  activeChart === 'lead' ? 'bg-blue-500 text-white' : 'bg-white text-black'
                }`}
                onClick={() => setActiveChart('lead')}
              >
                📆 Lead Time
              </button>
              <button
                className={`px-4 py-2 rounded-full text-sm ${
                  activeChart === 'duration' ? 'bg-blue-500 text-white' : 'bg-white text-black'
                }`}
                onClick={() => setActiveChart('duration')}
              >
                🕒 Duration
              </button>
            </div>

            <div className="bg-white text-black p-6 mt-4 rounded-xl shadow-lg">
              <Line data={activeChart === 'lead' ? leadChart : durationChart} />
            </div>

            <RouteMap origin={origin} destination={destination} />
          </>
        )}
      </div>
    </div>
  );
}
