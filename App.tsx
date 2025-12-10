import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
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
const ChatIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" /></svg>;
const SearchIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>;
const MapPinIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" /></svg>;
const DiceIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" /></svg>;


// --- Components ---

const PlaceCard: React.FC<{ place: Place }> = ({ place }) => {
  const [tip, setTip] = useState<string | null>(null);
  const [loadingTip, setLoadingTip] = useState(false);

  const handleGetTip = async () => {
    if (tip) return;
    setLoadingTip(true);
    try {
      const quickTip = await geminiService.getQuickTip(place.name);
      setTip(quickTip);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingTip(false);
    }
  };

  const getPriceColor = (level?: string) => {
    switch(level) {
      case 'High': return 'text-red-400';
      case 'Medium': return 'text-yellow-400';
      case 'Low': return 'text-green-400';
      default: return 'text-gray-400';
    }
  };

  const isOpenNow = place.isOpen;

  return (
    <div className="bg-gray-800 rounded-xl overflow-hidden shadow-lg border border-gray-700 flex flex-col h-full">
      <div className="h-40 bg-gray-700 relative">
        <img
          src={`https://placehold.co/600x400/374151/FFFFFF?text=${encodeURIComponent(place.name)}`}
          alt={place.name}
          className="w-full h-full object-cover"
        />
        <div className="absolute top-2 right-2 bg-black/60 px-2 py-1 rounded text-xs text-white font-bold backdrop-blur-sm">
          ⭐ {place.rating}
        </div>
        {place.isOpen !== undefined && (
          <div className={`absolute top-2 left-2 px-2 py-1 rounded text-xs text-white font-bold backdrop-blur-sm ${isOpenNow ? 'bg-green-600/80' : 'bg-red-600/80'}`}>
            {isOpenNow ? 'Open Now' : 'Closed'}
          </div>
        )}
      </div>
      <div className="p-4 flex-grow flex flex-col">
        <div className="flex justify-between items-start">
          <h3 className="text-xl font-bold text-white mb-1 leading-tight">{place.name}</h3>
        </div>
        <p className="text-sm text-blue-400 mb-2 font-medium">{place.category}</p>
        
        {/* Distance & Time Info */}
        {(place.distance || place.travelTime) && (
            <div className="flex flex-wrap gap-2 text-xs text-gray-300 mb-3 bg-gray-900/50 p-2 rounded-lg">
                {place.distance && (
                    <div className="flex items-center gap-1">
                        <span>📍</span>
                        <span>{place.distance}</span>
                    </div>
                )}
                {place.distance && place.travelTime && <span className="text-gray-600">•</span>}
                {place.travelTime && (
                    <div className="flex items-center gap-1">
                        <span>⏱️</span>
                        <span>{place.travelTime}</span>
                    </div>
                )}
            </div>
        )}

        {/* Menu & Price Info */}
        {(place.priceLevel || place.priceRange || place.popularDishes) && (
          <div className="mb-3 text-sm">
             <div className="flex items-center gap-2 mb-1">
                {place.priceLevel && <span className={`font-semibold ${getPriceColor(place.priceLevel)}`}>{place.priceLevel === 'Low' ? '$' : place.priceLevel === 'Medium' ? '$$' : '$$$'}</span>}
                {place.priceRange && <span className="text-gray-400 text-xs">({place.priceRange})</span>}
             </div>
             {place.popularDishes && (
               <div className="text-xs text-gray-400">
                 <span className="text-gray-500 font-semibold">Popular: </span>
                 {Array.isArray(place.popularDishes) ? place.popularDishes.join(', ') : place.popularDishes}
               </div>
             )}
          </div>
        )}

        <p className="text-gray-300 text-sm mb-4 line-clamp-3 flex-grow">{place.description}</p>
        
        {place.openingHours && (
            <p className="text-xs text-gray-500 mb-3">🕒 {place.openingHours}</p>
        )}

        <div className="mt-auto space-y-2">
          {place.uri && (
            <a
              href={place.uri}
              target="_blank"
              rel="noopener noreferrer"
              className="block w-full text-center bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold py-2 rounded-lg transition-colors"
            >
              View Details
            </a>
          )}
          <button
            onClick={handleGetTip}
            disabled={loadingTip || !!tip}
            className="block w-full text-center bg-gray-700 hover:bg-gray-600 text-white text-sm font-semibold py-2 rounded-lg transition-colors disabled:opacity-50"
          >
            {loadingTip ? '...' : tip ? '💡 Tip Loaded' : '✨ Get a Local Tip'}
          </button>
          {tip && (
            <div className="bg-yellow-900/30 border border-yellow-700/50 p-2 rounded text-xs text-yellow-200 mt-2 italic">
              "{tip}"
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const HomePage = ({ 
  onSearch, 
  loading,
  location,
  setLocation,
  onGetLocation 
}: { 
  onSearch: (q: string) => void, 
  loading: boolean,
  location: {name?: string},
  setLocation: (loc: {name?: string}) => void,
  onGetLocation: () => void
}) => {
  const [inputValue, setInputValue] = useState('');
  const [mood, setMood] = useState<string | null>(null);

  const categories = [
    { name: 'Restaurants', icon: '🍽️' },
    { name: 'Clubs', icon: '🎶' },
    { name: 'Cafés', icon: '☕' },
    { name: 'Attractions', icon: '🌄' },
    { name: 'Events', icon: '🎉' },
    { name: 'Adventures', icon: '🧭' },
    { name: 'Hotels', icon: '🏨' },
  ];

  const moods = [
    { name: 'Romantic', icon: '🌹' },
    { name: 'Chill', icon: '😌' },
    { name: 'Energetic', icon: '⚡' },
    { name: 'Hidden Gem', icon: '💎' },
  ];

  const handleSearch = () => {
    if (inputValue.trim()) {
      let query = inputValue;
      if (mood) query += ` (${mood} vibe)`;
      onSearch(query);
    }
  };

  const handleSurpriseMe = () => {
      onSearch("Unique hidden gem with high ratings, surprise me");
  };

  return (
    <div className="p-6 pb-24">
      <header className="mb-8 mt-4">
        <h1 className="text-4xl font-extrabold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-teal-400">Spotly</h1>
        <p className="text-gray-400">Discover your world, one spot at a time.</p>
      </header>

      <div className="space-y-6">
        {/* Location Section */}
        <div className="bg-gray-800 p-4 rounded-xl border border-gray-700 shadow-sm">
            <label className="block text-sm font-medium text-gray-400 mb-2">📍 Where are you?</label>
            <div className="flex gap-2">
                <input
                    type="text"
                    placeholder="Enter city or area..."
                    className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                    value={location.name || ''}
                    onChange={(e) => setLocation({ name: e.target.value })}
                />
                <button 
                    onClick={onGetLocation}
                    className="bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-lg transition-colors flex items-center justify-center"
                    title="Use Current Location"
                >
                    <MapPinIcon />
                </button>
            </div>
        </div>

        {/* Search Section */}
        <div className="bg-gray-800 p-4 rounded-xl border border-gray-700 shadow-sm">
            <label className="block text-sm font-medium text-gray-400 mb-2">What are you looking for?</label>
            
            {/* Mood Selector */}
            <div className="flex gap-2 mb-4 overflow-x-auto pb-2 scrollbar-hide">
                {moods.map((m) => (
                    <button
                        key={m.name}
                        onClick={() => setMood(mood === m.name ? null : m.name)}
                        className={`flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-medium whitespace-nowrap transition-colors border ${
                            mood === m.name 
                            ? 'bg-purple-600 border-purple-500 text-white' 
                            : 'bg-gray-900 border-gray-700 text-gray-400 hover:border-gray-500'
                        }`}
                    >
                        <span>{m.icon}</span>
                        {m.name}
                    </button>
                ))}
            </div>

            <div className="relative">
                <input
                    type="text"
                    placeholder="Type a category (e.g., Pizza) or choose below..."
                    className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 pl-10 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                />
                <div className="absolute left-3 top-3.5 text-gray-500">
                    <SearchIcon />
                </div>
            </div>

            <div className="grid grid-cols-2 gap-3 mt-4">
                {categories.map((cat) => (
                    <button
                        key={cat.name}
                        onClick={() => {
                            setInputValue(cat.name);
                            onSearch(cat.name + (mood ? ` ${mood} vibe` : ''));
                        }}
                        className="bg-gray-700 hover:bg-gray-600 p-3 rounded-lg text-left transition-colors flex items-center gap-2"
                    >
                        <span className="text-xl">{cat.icon}</span>
                        <span className="font-medium">{cat.name}</span>
                    </button>
                ))}
            </div>

            {/* Surprise Me Button */}
            <button
                onClick={handleSurpriseMe}
                className="w-full mt-4 bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-500 hover:to-purple-500 text-white py-3 rounded-lg font-bold shadow-lg transform active:scale-95 transition-all flex items-center justify-center gap-2"
            >
                <DiceIcon />
                Surprise Me!
            </button>
        </div>
        
        <button
            onClick={handleSearch}
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-500 text-white py-4 rounded-xl font-bold text-lg shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all active:scale-95"
        >
            {loading ? 'Finding Places...' : 'Find Places ➡️'}
        </button>

         {/* Disclaimer */}
         <div className="mt-8 px-2">
            <p className="text-[10px] text-gray-500 text-center leading-relaxed">
              Information shown in Spotly comes from multiple public sources and partner platforms.
              Spotly acts as a smart bridge to help you discover places easily — but some details like menus, prices, or opening hours may change without notice.
              Always double-check important information directly with the place before making plans.
            </p>
        </div>
      </div>
    </div>
  );
};

const ResultsView = ({ 
    places, 
    onBack, 
    groundingChunks,
    onLoadMore,
    loadingMore
}: { 
    places: Place[], 
    onBack: () => void, 
    groundingChunks: GroundingChunk[],
    onLoadMore: () => void,
    loadingMore: boolean
}) => {
    // Pagination state
    const [currentPage, setCurrentPage] = useState(1);
    const [filterOpen, setFilterOpen] = useState(false);
    const itemsPerPage = 6;

    // Filter state
    const [onlyOpen, setOnlyOpen] = useState(false);
    const [minRating, setMinRating] = useState(0);

    const filteredPlaces = places.filter(p => {
        if (onlyOpen && p.isOpen === false) return false;
        if (p.rating < minRating) return false;
        return true;
    });

    const totalPages = Math.ceil(filteredPlaces.length / itemsPerPage);
    
    // Get current page items
    const currentPlaces = filteredPlaces.slice(
        (currentPage - 1) * itemsPerPage, 
        currentPage * itemsPerPage
    );

    const scrollToTop = () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    const handleNextPage = () => {
        if (currentPage < totalPages) {
            setCurrentPage(p => p + 1);
            scrollToTop();
        }
    };

    const handlePrevPage = () => {
        if (currentPage > 1) {
            setCurrentPage(p => p - 1);
            scrollToTop();
        }
    };

    return (
        <div className="p-4 pb-24 min-h-screen">
            <div className="flex items-center gap-4 mb-6 sticky top-0 bg-gray-900/95 backdrop-blur z-10 py-4 border-b border-gray-800">
                <button onClick={onBack} className="text-gray-400 hover:text-white">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" /></svg>
                </button>
                <h2 className="text-xl font-bold">Found Places</h2>
                <div className="ml-auto">
                    <button 
                        onClick={() => setFilterOpen(!filterOpen)}
                        className={`text-sm px-3 py-1 rounded-full border ${onlyOpen || minRating > 0 ? 'bg-blue-900 border-blue-500 text-blue-200' : 'border-gray-600 text-gray-400'}`}
                    >
                        Filters
                    </button>
                </div>
            </div>

            {filterOpen && (
                <div className="bg-gray-800 p-4 rounded-lg mb-6 border border-gray-700">
                    <div className="flex items-center justify-between mb-4">
                        <span className="text-sm font-medium">Open Now Only</span>
                        <button 
                            onClick={() => setOnlyOpen(!onlyOpen)}
                            className={`w-10 h-6 rounded-full p-1 transition-colors ${onlyOpen ? 'bg-blue-600' : 'bg-gray-600'}`}
                        >
                            <div className={`w-4 h-4 bg-white rounded-full shadow-md transform transition-transform ${onlyOpen ? 'translate-x-4' : ''}`} />
                        </button>
                    </div>
                     <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Min Rating: {minRating > 0 ? `${minRating}+` : 'Any'}</span>
                        <div className="flex gap-1">
                            {[0, 3, 4, 4.5].map(r => (
                                <button 
                                    key={r}
                                    onClick={() => setMinRating(r)}
                                    className={`px-2 py-1 text-xs rounded ${minRating === r ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-400'}`}
                                >
                                    {r === 0 ? 'All' : r}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {currentPlaces.map((place, index) => (
                    <PlaceCard key={index} place={place} />
                ))}
            </div>

            {filteredPlaces.length === 0 && (
                <div className="text-center py-20 text-gray-500">
                    <p className="text-lg">No places found with current filters.</p>
                    <button onClick={() => {setOnlyOpen(false); setMinRating(0);}} className="text-blue-400 mt-2 hover:underline">Clear filters</button>
                </div>
            )}

            {/* Pagination Controls */}
            {filteredPlaces.length > 0 && (
                <div className="flex flex-col gap-4 mt-8">
                     <div className="flex justify-center items-center gap-4">
                        <button
                            onClick={handlePrevPage}
                            disabled={currentPage === 1}
                            className="px-4 py-2 bg-gray-800 rounded-lg text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-700 transition-colors"
                        >
                            Previous
                        </button>
                        <span className="text-sm text-gray-400">
                            Page <span className="text-white font-bold">{currentPage}</span> of {totalPages}
                        </span>
                        <button
                            onClick={handleNextPage}
                            disabled={currentPage === totalPages}
                            className="px-4 py-2 bg-gray-800 rounded-lg text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-700 transition-colors"
                        >
                            Next
                        </button>
                    </div>
                    
                    {/* Search More Button */}
                    <button
                        onClick={onLoadMore}
                        disabled={loadingMore}
                        className="w-full bg-gray-800 hover:bg-gray-700 border border-gray-700 text-white py-3 rounded-lg font-medium text-sm transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                    >
                        {loadingMore ? (
                            <>
                                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                                Finding More Places...
                            </>
                        ) : (
                            <>
                                <SearchIcon />
                                Search More Places
                            </>
                        )}
                    </button>
                </div>
            )}

            {groundingChunks.length > 0 && (
                <div className="mt-8 pt-6 border-t border-gray-800">
                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Sources</h4>
                    <div className="flex flex-wrap gap-2">
                        {groundingChunks.map((chunk, i) => (
                            <div key={i}>
                                {chunk.web?.uri && (
                                    <a href={chunk.web.uri} target="_blank" rel="noopener noreferrer" className="text-xs text-blue-500 hover:underline bg-blue-900/20 px-2 py-1 rounded block truncate max-w-xs">
                                        {chunk.web.title || new URL(chunk.web.uri).hostname}
                                    </a>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

// ... (ProPlanView kept simple for brevity, standard implementation)
const ProPlanView = () => (
    <div className="p-6 pb-24 flex flex-col items-center justify-center min-h-[80vh] text-center">
        <div className="w-16 h-16 bg-gradient-to-br from-yellow-400 to-orange-500 rounded-2xl flex items-center justify-center mb-6 shadow-orange-500/20 shadow-lg">
            <span className="text-3xl text-white">👑</span>
        </div>
        <h2 className="text-2xl font-bold text-white mb-2">Spotly Pro</h2>
        <p className="text-gray-400 mb-8 max-w-sm">Unlock AI-powered itineraries, offline maps, and exclusive local deals.</p>
        <div className="bg-gray-800 p-6 rounded-xl w-full max-w-xs border border-gray-700 mb-6 relative overflow-hidden">
            <div className="absolute top-0 right-0 bg-blue-600 text-[10px] font-bold px-2 py-1 rounded-bl">EARLY BIRD</div>
            <p className="text-3xl font-bold text-white mb-1">$4.99<span className="text-sm text-gray-500 font-normal">/mo</span></p>
            <ul className="text-left text-sm text-gray-300 space-y-2 mt-4">
                <li className="flex items-center gap-2">✅ Unlimited AI Itineraries</li>
                <li className="flex items-center gap-2">✅ Ad-free Experience</li>
                <li className="flex items-center gap-2">✅ Priority Support</li>
            </ul>
        </div>
        <button className="bg-white text-gray-900 font-bold py-3 px-8 rounded-full hover:bg-gray-100 transition-colors">
            Upgrade Now
        </button>
    </div>
);

// --- Chat Overlay ---
const ChatOverlay = ({ onClose }: { onClose: () => void }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const chatRef = useRef<Chat | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    chatRef.current = geminiService.createChat();
    // Initialize welcome message
    setMessages([{ role: 'model', text: "Hi! I'm Spotly Assistant. Need help finding a place or planning a trip?" }]);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const playAudio = async (text: string) => {
    try {
        const base64Audio = await geminiService.getTTSAudio(text);
        if (!audioContextRef.current) {
            audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({sampleRate: 24000});
        }
        const ctx = audioContextRef.current;
        const audioBuffer = await decodeAudioData(decode(base64Audio), ctx, 24000, 1);
        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(ctx.destination);
        source.start();
    } catch (e) {
        console.error("Audio playback failed", e);
    }
  }

  const handleSend = async () => {
    if (!input.trim() || !chatRef.current) return;
    
    const userMsg: ChatMessage = { role: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      // FIX: sendMessage accepts an object with a 'message' property, not a string directly.
      const result = await chatRef.current.sendMessage({ message: userMsg.text });
      // FIX: The response text is available directly on the result object as 'text', not 'result.response.text'.
      const modelResponse = result.text;
      setMessages(prev => [...prev, { role: 'model', text: modelResponse || '' }]);
      
      // Attempt to read short responses
      if (modelResponse && modelResponse.length < 200) {
          playAudio(modelResponse);
      }

    } catch (e) {
      console.error(e);
      setMessages(prev => [...prev, { role: 'model', text: "Sorry, I'm having trouble connecting right now." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-gray-900 z-50 flex flex-col">
      <div className="p-4 border-b border-gray-800 flex items-center justify-between bg-gray-900">
        <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center">
                <ChatIcon />
            </div>
            <h3 className="font-bold text-white">Spotly Assistant</h3>
        </div>
        <button onClick={onClose} className="text-gray-400 hover:text-white p-2">✕</button>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] rounded-2xl px-4 py-3 ${
              msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-200'
            }`}>
              {msg.text}
            </div>
          </div>
        ))}
        {loading && (
             <div className="flex justify-start">
                <div className="bg-gray-800 rounded-2xl px-4 py-3 flex gap-1">
                    <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></span>
                    <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce delay-100"></span>
                    <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce delay-200"></span>
                </div>
             </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 bg-gray-900 border-t border-gray-800">
        <div className="flex gap-2">
          <input 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask anything..." 
            className="flex-1 bg-gray-800 border-none rounded-full px-4 py-3 text-white focus:ring-2 focus:ring-blue-600"
          />
          <button 
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="bg-blue-600 text-white p-3 rounded-full hover:bg-blue-700 disabled:opacity-50"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 transform rotate-90" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>
          </button>
        </div>
      </div>
    </div>
  );
};


// --- Main App Component ---

const App = () => {
  const [view, setView] = useState<'home' | 'results' | 'pro'>('home');
  const [places, setPlaces] = useState<Place[]>([]);
  const [groundingChunks, setGroundingChunks] = useState<GroundingChunk[]>([]);
  const [loading, setLoading] = useState(false);
  const [location, setLocation] = useState<{ lat?: number; lon?: number; name?: string }>({});
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [currentQuery, setCurrentQuery] = useState('');
  const [loadingMore, setLoadingMore] = useState(false);
  
  // State to track if inputs are focused to hide nav
  const [isInputFocused, setIsInputFocused] = useState(false);

  // Global focus listener
  useEffect(() => {
    const handleFocus = (e: FocusEvent) => {
        if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'TEXTAREA') {
            setIsInputFocused(true);
        }
    };
    const handleBlur = (e: FocusEvent) => {
        if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'TEXTAREA') {
            setIsInputFocused(false);
        }
    };

    window.addEventListener('focus', handleFocus, true);
    window.addEventListener('blur', handleBlur, true);
    return () => {
        window.removeEventListener('focus', handleFocus, true);
        window.removeEventListener('blur', handleBlur, true);
    };
  }, []);

  const handleGetLocation = () => {
    if (!navigator.geolocation) {
        alert("Geolocation is not supported by this browser.");
        return;
    }
    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords;
        // Optional: Reverse geocode to get city name if needed, but for now we just store coords
        // For better UX, we might want to query Gemini to get the city name from lat/long,
        // but passing lat/long to geminiService is sufficient for search.
        setLocation({ lat: latitude, lon: longitude, name: "Current Location" });
      },
      (error) => {
        console.error("Error getting location:", error);
        if (error.code === error.PERMISSION_DENIED) {
            alert("Please enable location permissions in your browser settings to use this feature.");
        } else {
            alert("Unable to retrieve your location.");
        }
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
    );
  };

  const handleSearch = async (query: string) => {
    if (!location.name && !location.lat) {
        alert("Please enter a location or use current location.");
        return;
    }

    setLoading(true);
    setCurrentQuery(query);
    try {
      const { places: fetchedPlaces, groundingChunks: fetchedChunks } = await geminiService.getPlaceRecommendations(query, location);
      setPlaces(fetchedPlaces);
      setGroundingChunks(fetchedChunks);
      setView('results');
    } catch (error) {
      console.error(error);
      alert("Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleLoadMore = async () => {
    if (!currentQuery) return;
    setLoadingMore(true);
    try {
      // Create a list of names to exclude to avoid duplicates
      const excludeNames = places.map(p => p.name);
      const { places: newPlaces, groundingChunks: newChunks } = await geminiService.getPlaceRecommendations(currentQuery, location, excludeNames);
      
      // Append new unique places
      setPlaces(prev => [...prev, ...newPlaces]);
      setGroundingChunks(prev => [...prev, ...newChunks]);
    } catch (e) {
       console.error(e);
       alert("Could not load more places.");
    } finally {
       setLoadingMore(false);
    }
  };

  return (
    <div className="min-h-screen font-sans bg-gray-900 text-white relative">
      {/* Main Content Area */}
      <main className="max-w-md mx-auto min-h-screen bg-gray-900 relative shadow-2xl overflow-hidden">
        {view === 'home' && (
          <HomePage 
            onSearch={handleSearch} 
            loading={loading} 
            location={location}
            setLocation={setLocation}
            onGetLocation={handleGetLocation}
          />
        )}
        {view === 'results' && (
          <ResultsView 
            places={places} 
            onBack={() => setView('home')} 
            groundingChunks={groundingChunks}
            onLoadMore={handleLoadMore}
            loadingMore={loadingMore}
          />
        )}
        {view === 'pro' && <ProPlanView />}

        {/* Floating Chat Button (Only visible if chat is closed and nav is visible) */}
        {!isChatOpen && !isInputFocused && (
          <button 
            onClick={() => setIsChatOpen(true)}
            className="fixed bottom-24 right-6 bg-blue-600 hover:bg-blue-500 text-white p-4 rounded-full shadow-lg z-30 transition-transform active:scale-90"
          >
            <ChatIcon />
          </button>
        )}
        
        {/* Chat Overlay */}
        {isChatOpen && <ChatOverlay onClose={() => setIsChatOpen(false)} />}

        {/* Bottom Navigation (Hidden when typing or chat is open) */}
        {!isChatOpen && !isInputFocused && (
          <nav className="fixed bottom-0 left-0 right-0 max-w-md mx-auto bg-gray-900/90 backdrop-blur border-t border-gray-800 flex justify-around py-4 z-20 pb-8">
            <button 
              onClick={() => setView('home')}
              className={`flex flex-col items-center gap-1 ${view === 'home' ? 'text-blue-500' : 'text-gray-500'}`}
            >
              <CompassIcon />
              <span className="text-xs font-medium">Home</span>
            </button>
            <button 
              onClick={() => setView('pro')}
              className={`flex flex-col items-center gap-1 ${view === 'pro' ? 'text-blue-500' : 'text-gray-500'}`}
            >
              <PlanIcon />
              <span className="text-xs font-medium">Pro Plan</span>
            </button>
          </nav>
        )}
      </main>
    </div>
  );
};

export default App;