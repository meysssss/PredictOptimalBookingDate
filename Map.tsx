import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

interface Props {
  origin: string;
  destination: string;
}

const airportCoords: Record<string, [number, number]> = {
  YYZ: [43.6777, -79.6248],
  YVR: [49.1947, -123.1792],
  YUL: [45.4706, -73.7408],
  YYC: [51.1139, -114.0203],
  YEG: [53.3097, -113.5796],
  BCN: [41.2974, 2.0833],
  FCO: [41.7999, 12.2462],
  ZRH: [47.4647, 8.5492],
  LIS: [38.7742, -9.1342],
  BRU: [50.9014, 4.4844],
  JFK: [40.6413, -73.7781],
  ORD: [41.9742, -87.9073],
  LAX: [33.9416, -118.4085],
  SFO: [37.6213, -122.3790]
};

const icon = new L.Icon({
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});

export default function RouteMap({ origin, destination }: Props) {
  const orig = airportCoords[origin];
  const dest = airportCoords[destination];
  if (!orig || !dest) return null;

  return (
    <div className="h-[400px] w-full mt-6 rounded-xl overflow-hidden">
      <MapContainer bounds={[orig, dest]} scrollWheelZoom={false} className="h-full w-full">
        <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" />
        <Marker position={orig} icon={icon}>
          <Popup>Origin: {origin}</Popup>
        </Marker>
        <Marker position={dest} icon={icon}>
          <Popup>Destination: {destination}</Popup>
        </Marker>
        <Polyline positions={[orig, dest]} color="#3b82f6" />
      </MapContainer>
    </div>
  );
}
