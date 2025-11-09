import React, { useState, useMemo, useCallback, useRef } from 'react';
import { Place, AspectRatio, ChatMessage, GroundingChunk } from './types';
import * as geminiService from './services/geminiService';
import { Chat, GenerateContentResponse } from '@google/genai';

// --- Helper Functions for Audio ---
function decode(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}


// --- SVG Icons ---
const CompassIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 21a9 9 0 100-18 9 9 0 000 18z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m8.66-13.66l-.7.7M4.04 19.96l-.7.7M21 12h-1m-16 0H3m13.66 8.66l-.7-.7m-11.22-11.22l-.7-.7" /></svg>;
const PlanIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l5.447 2.724A1 1 0 0021 16.382V5.618a1 1 0 00-1.447-.894L15 7m-6 10v-5m6 5v-5m0 0l-6-3" /></svg>;
const ChatIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" /></svg>;
const StarIcon: React.FC<{ filled: boolean }> = ({ filled }) => <svg xmlns="http://www.w3.org/2000/svg" className={`h-5 w-5 ${filled ? 'text-yellow-400' : 'text-gray-600'}`} viewBox="0 0 20 20" fill="currentColor"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" /></svg>;
const HeartIcon = ({ filled }: { filled: boolean }) => <svg xmlns="http://www.w3.org/2000/svg" className={`h-6 w-6 transition-colors duration-200 ${filled ? 'text-red-500' : 'text-gray-400 hover:text-red-400'}`} viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clipRule="evenodd" /></svg>;
const SpeakerIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.707.707L4.586 13H2a1 1 0 01-1-1V8a1 1 0 011-1h2.586l3.707-3.707a1 1 0 011.09-.217zM14.657 2.929a1 1 0 011.414 0A9.972 9.972 0 0119 10a9.972 9.972 0 01-2.929 7.071 1 1 0 01-1.414-1.414A7.971 7.971 0 0017 10c0-2.21-.894-4.208-2.343-5.657a1 1 0 010-1.414zm-2.829 2.828a1 1 0 011.415 0A5.983 5.983 0 0115 10a5.984 5.984 0 01-1.757 4.243 1 1 0 01-1.415-1.415A3.984 3.984 0 0013 10a3.983 3.983 0 00-1.172-2.828 1 1 0 010-1.414z" clipRule="evenodd" /></svg>;
const CloseIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>;
const SendIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-8.707l-3-3a1 1 0 00-1.414 1.414L10.586 9H7a1 1 0 100 2h3.586l-1.293 1.293a1 1 0 101.414 1.414l3-3a1 1 0 000-1.414z" clipRule="evenodd" /></svg>;
const ProfileIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.121 17.804A13.937 13.937 0 0112 16c2.5 0 4.847.655 6.879 1.804M15 10a3 3 0 11-6 0 3 3 0 016 0zm6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>;
const RestaurantIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 21v-8.172a4 4 0 00-1.172-2.828L5.172 4.343A2 2 0 016.586 3h10.828a2 2 0 011.414 1.343l-5.656 5.657A4 4 0 0012 12.828V21z" /></svg>;
const ClubIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.102 6.102a3.003 3.003 0 014.243 0L9 7.757l2.652-2.651a3.003 3.003 0 114.243 4.243L13.243 12l2.651 2.652a3.003 3.003 0 11-4.243 4.243L9 16.243l-1.652 1.651a3.003 3.003 0 01-4.243-4.243L5.757 12 3.102 9.348a3.003 3.003 0 010-4.243z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18 9.243l1.652-1.651a3.003 3.003 0 014.243 4.243L21.243 12l-2.651 2.652" /></svg>;
const CafeIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2zM6 11V7a2 2 0 012-2h8a2 2 0 012 2v4" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5V3m4 2V3m4 2V3" /></svg>;
const AttractionIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4-1.79-4-4-4zm0 0V4m0 16v-4m8-8h-4M4 12h4m10.485-5.515l-2.829 2.829m-8.342 8.342l-2.829 2.829M5.515 5.515l2.829 2.829m8.342 8.342l2.829 2.829" /></svg>;
const EventIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>;
const AdventureIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 21v-4m0 0V5a2 2 0 012-2h6.5l1 1H21l-3 6 3 6H8.5l-1-1H5a2 2 0 00-2 2zm9-13.5V6" /></svg>;
const ShareIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path d="M15 8a3 3 0 10-2.977-2.63l-4.94 2.47a3 3 0 100 4.319l4.94 2.47a3 3 0 10.895-1.789l-4.94-2.47a3.027 3.027 0 000-.74l4.94-2.47C13.456 7.68 14.19 8 15 8z" /></svg>;


// --- Main App Component ---
export default function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Discovery state
  const [places, setPlaces] = useState<Place[]>([]);
  const [favorites, setFavorites] = useState<Place[]>([]);
  const [groundingChunks, setGroundingChunks] = useState<GroundingChunk[]>([]);
  const [isShowingResults, setIsShowingResults] = useState(false);

  // Pro Planner state
  const [itinerary, setItinerary] = useState('');

  // Chatbot state
  const [isChatOpen, setIsChatOpen] = useState(false);
  const chatRef = useRef<Chat | null>(null);

  const toggleFavorite = (place: Place) => {
    setFavorites(prev =>
      prev.find(p => p.name === place.name)
        ? prev.filter(p => p.name !== place.name)
        : [...prev, place]
    );
  };

  const handleDiscover = async (query: string, location: { lat?: number; lon?: number; name?: string }) => {
    setIsLoading(true);
    setError(null);
    setPlaces([]);
    setGroundingChunks([]);

    try {
      const result = await geminiService.getPlaceRecommendations(query, location);
      setPlaces(result.places);
      setGroundingChunks(result.groundingChunks);
      setIsShowingResults(true);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleProPlanner = async () => {
    if (favorites.length === 0) {
      setError("Add some places to your favorites to create an itinerary.");
      return;
    }
    setIsLoading(true);
    setError(null);
    setItinerary('');
    try {
      const result = await geminiService.createProItinerary(favorites);
      setItinerary(result);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'home': return <HomePage onDiscover={handleDiscover} places={places} favorites={favorites} toggleFavorite={toggleFavorite} groundingChunks={groundingChunks} isShowingResults={isShowingResults} setIsShowingResults={setIsShowingResults}/>;
      case 'plan': return <ProPlannerTab onPlan={handleProPlanner} result={itinerary} favoritesCount={favorites.length} />;
      default: return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col font-sans">
      <Header />
      <main className="flex-grow container mx-auto p-4 sm:p-6 pb-24">
        {error && <div className="bg-red-500/20 border border-red-500 text-red-300 p-3 rounded-lg mb-4">{error}</div>}
        {isLoading && <LoadingSpinner />}
        
        <div className="bg-gray-800/50 rounded-xl shadow-2xl p-4 sm:p-6 backdrop-blur-sm">
          {renderContent()}
        </div>
      </main>

      <ChatBot isChatOpen={isChatOpen} setIsChatOpen={setIsChatOpen} chatRef={chatRef} />

      <nav className="fixed bottom-0 left-0 right-0 bg-gray-900/80 backdrop-blur-lg border-t border-t-cyan-400/20 flex justify-around p-2 shadow-t-xl">
        <NavButton icon={<CompassIcon />} label="Home" isActive={activeTab === 'home'} onClick={() => {setActiveTab('home'); setIsShowingResults(false);}} />
        <NavButton icon={<PlanIcon />} label="Pro Plan" isActive={activeTab === 'plan'} onClick={() => setActiveTab('plan')} />
      </nav>
    </div>
  );
}

// --- Sub-components for Tabs and UI Elements ---

const Header = () => (
  <header className="p-4 flex justify-between items-center">
    <div className="w-6"></div> {/* Spacer */}
    <div className="text-center">
        <h1 className="text-4xl sm:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500">
          Spotly
        </h1>
        <p className="text-gray-400 mt-1">Discover Your Next Adventure</p>
    </div>
    <button className="text-gray-400 hover:text-white transition-colors">
        <ProfileIcon />
    </button>
  </header>
);

const NavButton: React.FC<{ icon: React.ReactNode; label: string; isActive: boolean; onClick: () => void }> = ({ icon, label, isActive, onClick }) => (
  <button onClick={onClick} className={`flex flex-col items-center justify-center w-20 h-16 rounded-lg transition-all duration-300 ${isActive ? 'bg-cyan-500/20 text-cyan-400 scale-110' : 'text-gray-400 hover:bg-gray-700/50 hover:text-white'}`}>
    {icon}
    <span className="text-xs mt-1">{label}</span>
  </button>
);

const LoadingSpinner = () => (
  <div className="fixed inset-0 bg-gray-900/50 flex justify-center items-center z-50">
    <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-cyan-500"></div>
  </div>
);

const HomePage: React.FC<{
  onDiscover: (query: string, location: { lat?: number; lon?: number; name?: string }) => void;
  places: Place[];
  favorites: Place[];
  toggleFavorite: (place: Place) => void;
  groundingChunks: GroundingChunk[];
  isShowingResults: boolean;
  setIsShowingResults: (isShowing: boolean) => void;
}> = ({ onDiscover, places, favorites, toggleFavorite, groundingChunks, isShowingResults, setIsShowingResults }) => {
  const [selectedCategory, setSelectedCategory] = useState('Restaurants');
  const [customQuery, setCustomQuery] = useState('');
  const [locationStatus, setLocationStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [manualLocation, setManualLocation] = useState('');
  const [coords, setCoords] = useState<{lat: number, lon: number} | null>(null);

  const handleGetLocation = () => {
    setLocationStatus('loading');
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setCoords({ lat: position.coords.latitude, lon: position.coords.longitude });
        setLocationStatus('success');
      },
      () => setLocationStatus('error')
    );
  };
  
  const handleSearch = () => {
    const query = customQuery.trim() || selectedCategory;
    if (manualLocation.trim()) {
      onDiscover(query, { name: manualLocation.trim() });
    } else if (coords) {
      onDiscover(query, { lat: coords.lat, lon: coords.lon });
    }
  };

  const categories = [
    { name: 'Restaurants', icon: <RestaurantIcon /> },
    { name: 'Clubs', icon: <ClubIcon /> },
    { name: 'Cafés', icon: <CafeIcon /> },
    { name: 'Attractions', icon: <AttractionIcon /> },
    { name: 'Events', icon: <EventIcon /> },
    { name: 'Adventures', icon: <AdventureIcon /> },
  ];

  if (isShowingResults) {
    return <ResultsView 
              places={places} 
              favorites={favorites} 
              toggleFavorite={toggleFavorite} 
              groundingChunks={groundingChunks}
              onNewSearch={() => setIsShowingResults(false)}
            />;
  }

  return (
    <div className="flex flex-col items-center text-center max-w-2xl mx-auto">
        <div className="w-full bg-gray-800 p-6 rounded-xl shadow-lg mb-8">
            <h2 className="text-2xl font-bold mb-4 text-white">📍 Where are you?</h2>
            <div className="flex flex-col sm:flex-row gap-4">
                <button 
                    onClick={handleGetLocation} 
                    className="flex-1 bg-gray-700 hover:bg-gray-600 text-white font-semibold py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                    {locationStatus === 'loading' && <div className="w-5 h-5 border-2 border-t-transparent border-white rounded-full animate-spin"></div>}
                    {locationStatus === 'idle' && 'Use My Current Location'}
                    {locationStatus === 'success' && '✅ Location Ready'}
                    {locationStatus === 'error' && '❌ Error Getting Location'}
                </button>
                <input 
                    type="text" 
                    placeholder="Type your city or area..." 
                    value={manualLocation}
                    onChange={(e) => setManualLocation(e.target.value)}
                    className="flex-1 bg-gray-700 border border-gray-600 rounded-lg p-3 focus:ring-2 focus:ring-cyan-500 focus:outline-none" 
                />
            </div>
        </div>

        <div className="w-full mb-8">
            <h2 className="text-2xl font-bold mb-4 text-white">🔍 What are you looking for today?</h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-6">
                {categories.map(cat => (
                    <button 
                        key={cat.name} 
                        onClick={() => setSelectedCategory(cat.name)}
                        className={`p-4 rounded-xl shadow-md transition-all duration-200 flex flex-col items-center justify-center gap-2 ${selectedCategory === cat.name ? 'bg-cyan-500 text-white ring-2 ring-cyan-300 scale-105' : 'bg-gray-800 hover:bg-gray-700'}`}
                    >
                        {cat.icon}
                        <span className="font-semibold">{cat.name}</span>
                    </button>
                ))}
            </div>
            <input 
                type="text" 
                value={customQuery}
                onChange={(e) => setCustomQuery(e.target.value)}
                placeholder="Or write what you’re looking for... e.g. romantic dinner" 
                className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 focus:ring-2 focus:ring-cyan-500 focus:outline-none"
            />
        </div>

        <button 
            onClick={handleSearch} 
            disabled={locationStatus !== 'success' && !manualLocation.trim()}
            className="w-full max-w-sm bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 text-white font-bold py-4 px-8 rounded-xl transition-all duration-200 transform hover:scale-105 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
        >
            🔎 Find Places
        </button>
    </div>
  );
};

const ResultsView: React.FC<{
  places: Place[];
  favorites: Place[];
  toggleFavorite: (place: Place) => void;
  groundingChunks: GroundingChunk[];
  onNewSearch: () => void;
}> = ({ places, favorites, toggleFavorite, groundingChunks, onNewSearch }) => {
    return (
        <div>
            <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-cyan-400">Your Spots</h2>
                <button onClick={onNewSearch} className="bg-gray-700 hover:bg-gray-600 text-white font-bold py-2 px-4 rounded-lg transition-colors">
                    New Search
                </button>
            </div>

            {places.length === 0 && <p className="text-center text-gray-400">No places found. Try a different search!</p>}
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {places.map(place => (
                  <PlaceCard key={place.name} place={place} isFavorite={favorites.some(p => p.name === place.name)} onToggleFavorite={toggleFavorite} />
                ))}
            </div>

            {groundingChunks.length > 0 && (
                <div className="mt-8 p-4 bg-gray-800 rounded-lg">
                    <h3 className="text-lg font-semibold text-cyan-400 mb-2">Sources from Google Maps</h3>
                    <ul className="list-disc list-inside text-sm text-gray-300">
                        {groundingChunks.filter(c => c.maps).map((chunk, index) => (
                            <li key={index}>
                                <a href={chunk.maps?.uri} target="_blank" rel="noopener noreferrer" className="underline hover:text-cyan-300">
                                    {chunk.maps?.title}
                                </a>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};

const PlaceCard: React.FC<{ place: Place; isFavorite: boolean; onToggleFavorite: (place: Place) => void; }> = ({ place, isFavorite, onToggleFavorite }) => {
  const [tip, setTip] = useState('');
  const [isTipLoading, setIsTipLoading] = useState(false);
  const [audioContext] = useState(() => new (window.AudioContext || (window as any).webkitAudioContext)());
  const [shareStatus, setShareStatus] = useState<'idle' | 'copied'>('idle');

  const handlePlayTTS = async (text: string) => {
      try {
          const audioB64 = await geminiService.getTTSAudio(text);
          const audioBytes = decode(audioB64);
          const audioBuffer = await decodeAudioData(audioBytes, audioContext, 24000, 1);
          const source = audioContext.createBufferSource();
          source.buffer = audioBuffer;
          source.connect(audioContext.destination);
          source.start();
      } catch (error) {
          console.error("Error playing TTS:", error);
      }
  };

  const handleShare = async () => {
    const shareData = {
      title: place.name,
      text: `Check out ${place.name}: ${place.description}`,
      url: place.uri || window.location.href,
    };
    if (navigator.share) {
        try {
            await navigator.share(shareData);
        } catch (error) {
            console.error('Error sharing:', error);
        }
    } else {
        try {
            await navigator.clipboard.writeText(shareData.url);
            setShareStatus('copied');
            setTimeout(() => setShareStatus('idle'), 2000);
        } catch (error) {
            console.error('Failed to copy:', error);
            alert("Could not copy link to clipboard.");
        }
    }
  };

  const handleGetTip = async () => {
    setIsTipLoading(true);
    setTip(await geminiService.getQuickTip(place.name));
    setIsTipLoading(false);
  };
  
  return (
    <div className="bg-gray-800 rounded-lg shadow-lg overflow-hidden transform hover:-translate-y-1 transition-transform duration-300">
      <div className="p-5">
        <div className="flex justify-between items-start">
          <div>
            <h3 className="text-xl font-bold text-white">{place.name}</h3>
            <p className="text-sm text-cyan-400">{place.category}</p>
          </div>
          <button onClick={() => onToggleFavorite(place)}>
            <HeartIcon filled={isFavorite} />
          </button>
        </div>
        <p className="text-gray-300 mt-2 text-sm">{place.description}</p>
        <div className="flex items-center mt-4">
          {[...Array(5)].map((_, i) => <StarIcon key={i} filled={i < place.rating} />)}
          <span className="text-gray-400 text-xs ml-2">({place.rating.toFixed(1)})</span>
        </div>
        {place.uri && (
           <a href={place.uri} target="_blank" rel="noopener noreferrer" className="inline-block mt-2 text-xs text-blue-400 hover:underline">View on Map</a>
        )}
        <div className="mt-4 flex flex-col gap-2">
            <div className="flex gap-2">
                <button onClick={() => handlePlayTTS(`${place.name}. ${place.description}`)} className="flex-1 text-xs bg-gray-700 hover:bg-gray-600 text-white py-1 px-2 rounded-md flex items-center justify-center gap-1">
                    <SpeakerIcon /> Read
                </button>
                <button onClick={handleGetTip} className="flex-1 text-xs bg-cyan-600 hover:bg-cyan-500 text-white py-1 px-2 rounded-md" disabled={isTipLoading}>
                  {isTipLoading ? '...' : 'Quick Tip'}
                </button>
                <button onClick={handleShare} className="flex-1 text-xs bg-gray-700 hover:bg-gray-600 text-white py-1 px-2 rounded-md flex items-center justify-center gap-1">
                  {shareStatus === 'idle' ? <><ShareIcon /> Share</> : 'Copied!'}
                </button>
            </div>
            {tip && <p className="text-xs bg-gray-700/50 p-2 rounded-md text-cyan-200 italic">"{tip}"</p>}
        </div>
      </div>
    </div>
  );
};

const ProPlannerTab: React.FC<{ onPlan: () => void; result: string; favoritesCount: number; }> = ({ onPlan, result, favoritesCount }) => (
  <div className="text-center">
    <h2 className="text-2xl font-bold mb-4 text-yellow-400">Pro Itinerary Planner</h2>
    <p className="text-gray-400 mb-6">Add places to your favorites, then let our Pro Planner create the perfect day for you.</p>
    <button onClick={onPlan} className="bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-400 hover:to-orange-400 text-white font-bold py-3 px-6 rounded-lg transition-transform duration-200 transform hover:scale-105" disabled={favoritesCount === 0}>Plan My Day ({favoritesCount} favs)</button>
    {result && <div className="mt-6 bg-gray-700/50 p-4 rounded-lg text-left whitespace-pre-wrap">{result}</div>}
  </div>
);

const ChatBot: React.FC<{ isChatOpen: boolean, setIsChatOpen: (isOpen: boolean) => void, chatRef: React.MutableRefObject<Chat | null> }> = ({ isChatOpen, setIsChatOpen, chatRef }) => {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    React.useEffect(scrollToBottom, [messages]);
    
    React.useEffect(() => {
        if (isChatOpen && !chatRef.current) {
            chatRef.current = geminiService.createChat();
        }
    }, [isChatOpen, chatRef]);

    const handleSend = async () => {
        if (!input.trim() || !chatRef.current) return;
        
        const userMessage: ChatMessage = { role: 'user', text: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsTyping(true);

        try {
            const response = await chatRef.current.sendMessage({ message: input });
            const modelMessage: ChatMessage = { role: 'model', text: response.text };
            
            const groundingChunks: GroundingChunk[] = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
            if(groundingChunks.length > 0) {
              const sourcesText = groundingChunks.map(c => `- [${c.web?.title || 'Source'}](${c.web?.uri})`).join('\n');
              modelMessage.text += `\n\n**Sources:**\n${sourcesText}`;
            }

            setMessages(prev => [...prev, modelMessage]);
        } catch (e) {
            const errorMessage: ChatMessage = { role: 'model', text: "Sorry, I encountered an error." };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsTyping(false);
        }
    };

    if (!isChatOpen) {
        return (
            <button onClick={() => setIsChatOpen(true)} className="fixed bottom-24 right-4 bg-gradient-to-r from-cyan-500 to-purple-500 p-4 rounded-full shadow-lg hover:scale-110 transition-transform">
                <ChatIcon />
            </button>
        );
    }

    return (
        <div className="fixed inset-0 bg-gray-900/70 z-40 flex justify-center items-center backdrop-blur-sm">
            <div className="bg-gray-800 rounded-2xl shadow-2xl w-full max-w-lg h-[80vh] flex flex-col mx-4">
                <div className="flex justify-between items-center p-4 border-b border-gray-700">
                    <h3 className="text-xl font-bold text-cyan-400">Spotly Assistant</h3>
                    <button onClick={() => setIsChatOpen(false)}><CloseIcon /></button>
                </div>
                <div className="flex-grow p-4 overflow-y-auto">
                    <div className="flex flex-col gap-4">
                        {messages.map((msg, index) => (
                            <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                <div className={`max-w-xs md:max-w-md p-3 rounded-2xl ${msg.role === 'user' ? 'bg-cyan-600 rounded-br-none' : 'bg-gray-700 rounded-bl-none'}`}>
                                    <p className="text-sm" dangerouslySetInnerHTML={{ __html: msg.text.replace(/\n/g, '<br />').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }}></p>

                                </div>
                            </div>
                        ))}
                         {isTyping && <div className="flex justify-start"><div className="bg-gray-700 p-3 rounded-2xl rounded-bl-none"><span className="animate-pulse">...</span></div></div>}
                        <div ref={messagesEndRef} />
                    </div>
                </div>
                <div className="p-4 border-t border-gray-700 flex gap-2">
                    <input type="text" value={input} onChange={e => setInput(e.target.value)} onKeyPress={e => e.key === 'Enter' && handleSend()} placeholder="Ask anything..." className="flex-grow bg-gray-700 border border-gray-600 rounded-full py-2 px-4 focus:ring-2 focus:ring-cyan-500 focus:outline-none"/>
                    <button onClick={handleSend} className="bg-cyan-500 p-2 rounded-full hover:bg-cyan-400 transition-colors"><SendIcon /></button>
                </div>
            </div>
        </div>
    );
};