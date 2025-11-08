import { GoogleGenAI, Type, GenerateContentResponse, Chat, Modality } from "@google/genai";
import { Place, AspectRatio, GroundingChunk } from '../types';

// Utility to convert File -> base64
const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve((reader.result as string).split(',')[1]);
    reader.onerror = error => reject(error);
  });
};

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });

export const getPlaceRecommendations = async (lat: number, lon: number, query: string): Promise<{ places: Place[], groundingChunks: GroundingChunk[] }> => {
  try {
    const prompt = `Find cool and interesting places near latitude ${lat} and longitude ${lon}. I'm looking for "${query}". Suggest a mix of places that fit this search. For each place, provide a name, a brief, enticing description, category (like 'Restaurant', 'Cafe', 'Park'), and a rating out of 5. Your response must be a single JSON object with a key "places" that is an array of place objects. Each place object must have keys: "name", "description", "category", and "rating". Do not add any text before or after the JSON object.`;

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
      config: {
        tools: [{ googleMaps: {} }],
      },
      toolConfig: {
        retrievalConfig: {
          latLng: { latitude: lat, longitude: lon }
        }
      }
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
      const mapChunk = groundingChunks.find(c => c.maps && c.maps.title.toLowerCase().includes(place.name.toLowerCase()));
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

export const analyzeImageVibe = async (imageFile: File): Promise<string> => {
  const base64Image = await fileToBase64(imageFile);
  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: {
      parts: [
        { text: "Describe the vibe of this place. Is it good for relaxing, working, a date night, or something else? Be descriptive and engaging." },
        { inlineData: { mimeType: imageFile.type, data: base64Image } }
      ]
    },
  });
  return response.text;
};

export const generateDreamImage = async (prompt: string, aspectRatio: AspectRatio): Promise<string> => {
    const response = await ai.models.generateImages({
        model: 'imagen-4.0-generate-001',
        prompt: `An artistic, beautiful photograph of: ${prompt}`,
        config: {
          numberOfImages: 1,
          outputMimeType: 'image/jpeg',
          aspectRatio: aspectRatio,
        },
    });
    const base64ImageBytes = response.generatedImages[0].image.imageBytes;
    return `data:image/jpeg;base64,${base64ImageBytes}`;
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