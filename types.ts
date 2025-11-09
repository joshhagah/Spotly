export interface Place {
  name: string;
  description: string;
  category: string;
  rating: number;
  uri?: string;
  title?: string;
}

export type AspectRatio = "1:1" | "16:9" | "9:16" | "4:3" | "3:4";

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
}

export interface GroundingChunk {
  web?: {
    // FIX: Made uri and title optional to match the type from @google/genai
    uri?: string;
    title?: string;
  };
  maps?: {
    // FIX: Made uri and title optional to match the type from @google/genai
    uri?: string;
    title?: string;
    // FIX: Changed placeAnswerSources from an array of objects to a single object to match the type from @google/genai.
    placeAnswerSources?: {
        reviewSnippets: {
            // FIX: Made uri and text optional to align with the library's type definition.
            uri?: string;
            text?: string;
        }[];
    }
  };
}
