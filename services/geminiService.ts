import { GoogleGenAI, Type, GenerateContentResponse, Chat, Modality } from "@google/genai";
import { Place, AspectRatio, GroundingChunk } from '../types';

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });

export const getPlaceRecommendations = async (query: string, location: { lat?: number; lon?: number; name?: string }): Promise<{ places: Place[], groundingChunks: GroundingChunk[] }> => {
  try {
    let locationString: string;
    let toolConfig: any = undefined;

    if (location.lat && location.lon) {
        locationString = `near latitude ${location.lat} and longitude ${location.lon}`;
        toolConfig = {
            retrievalConfig: {
                latLng: { latitude: location.lat, longitude: location.lon }
            }
        };
    } else if (location.name) {
        locationString = `in or near ${location.name}`;
    } else {
        throw new Error("Location information (either coordinates or a name) is required.");
    }

    const prompt = `Find cool and interesting places ${locationString}. I'm looking for "${query}". Suggest a mix of places that fit this search. For each place, provide a name, a brief, enticing description, category (like 'Restaurant', 'Cafe', 'Park'), and a rating out of 5. Your response must be a single JSON object with a key "places" that is an array of place objects. Each place object must have keys: "name", "description", "category", and "rating". Do not add any text before or after the JSON object.`;

    const config: any = {
        tools: [{ googleMaps: {} }],
    };

    if (toolConfig) {
        config.toolConfig = toolConfig;
    }

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
      config: config,
    });

    const groundingChunks: GroundingChunk[] = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
    
    let jsonStr = response.text.trim();
    const jsonMatch = jsonStr.match(/```(json)?\n?([\s\S]*?)\n?```/);
    if (jsonMatch && jsonMatch[2]) {
      jsonStr = jsonMatch[2];
    }
    const parsed = JSON.parse(jsonStr);

    // Merge grounding data into places
    const placesWithGrounding = parsed.places.map((place: Place, index: number) => {
      const mapChunk = groundingChunks.find(c => c.maps && c.maps.title && c.maps.title.toLowerCase().includes(place.name.toLowerCase()));
      if (mapChunk && mapChunk.maps) {
        return { ...place, uri: mapChunk.maps.uri, title: mapChunk.maps.title };
      }
      return place;
    });

    return { places: placesWithGrounding, groundingChunks };
  } catch (error) {
    console.error("Error fetching place recommendations:", error);
    throw new Error("Could not fetch recommendations. Please try again.");
  }
};

export const createProItinerary = async (places: Place[]): Promise<string> => {
    const placeNames = places.map(p => p.name).join(', ');
    const prompt = `I want to visit these places: ${placeNames}. Create a detailed, smart, and enjoyable full-day itinerary. Consider travel time between places, best times to visit, and suggest a logical order. Make it sound like a pro travel guide.`;
    
    const response = await ai.models.generateContent({
        model: "gemini-2.5-pro",
        contents: prompt,
        config: {
            thinkingConfig: { thinkingBudget: 32768 }
        }
    });
    return response.text;
};

export const getQuickTip = async (placeName: string): Promise<string> => {
    const response = await ai.models.generateContent({
        model: 'gemini-flash-lite-latest',
        contents: `Give me one quick, interesting tip or fact about ${placeName}. Keep it under 20 words.`,
    });
    return response.text;
};

export const createChat = (): Chat => {
    return ai.chats.create({
        model: 'gemini-2.5-flash',
        config: {
            systemInstruction: "You are a helpful and friendly travel assistant for the Spotly app. You can provide information, answer questions, and use Google Search to find up-to-date information.",
            tools: [{ googleSearch: {} }]
        }
    });
};

export const getTTSAudio = async (text: string): Promise<string> => {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-preview-tts",
      contents: [{ parts: [{ text: `Say this clearly and pleasantly: ${text}` }] }],
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: 'Kore' },
            },
        },
      },
    });
    return response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data as string;
};