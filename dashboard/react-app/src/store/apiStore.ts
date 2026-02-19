import { create } from 'zustand';
import axios from 'axios';

const API_HOST = 'http://localhost:26080';

interface ApiStore {
  checkApiHealth: () => Promise<boolean>;
  testEncode: (text: string, optimization: string) => Promise<any>;
  testEntities: (text: string) => Promise<any>;
  testAnalogy: (word1: string, word2: string, word3: string) => Promise<any>;
  startServices: () => Promise<any>;
  stopServices: () => Promise<any>;
  getStatus: () => Promise<any>;
}

export const useApiStore = create<ApiStore>(() => ({
  checkApiHealth: async () => {
    try {
      const response = await axios.get(`${API_HOST}/health`, { timeout: 2000 });
      return response.status === 200;
    } catch {
      return false;
    }
  },

  testEncode: async (text: string, optimization: string) => {
    try {
      const response = await axios.post(`${API_HOST}/api/encode`, {
        text,
        optimization
      });
      return response.data;
    } catch (error: any) {
      return { error: error.message };
    }
  },

  testEntities: async (text: string) => {
    try {
      const response = await axios.post(`${API_HOST}/api/entities`, { text });
      return response.data;
    } catch (error: any) {
      return { error: error.message };
    }
  },

  testAnalogy: async (word1: string, word2: string, word3: string) => {
    try {
      const response = await axios.post(`${API_HOST}/api/analogy`, {
        word1,
        word2,
        word3
      });
      return response.data;
    } catch (error: any) {
      return { error: error.message };
    }
  },

  startServices: async () => {
    try {
      const response = await axios.post('http://localhost:8501/api/services/start');
      return response.data;
    } catch (error: any) {
      return { error: error.message };
    }
  },

  stopServices: async () => {
    try {
      const response = await axios.post('http://localhost:8501/api/services/stop');
      return response.data;
    } catch (error: any) {
      return { error: error.message };
    }
  },

  getStatus: async () => {
    try {
      const response = await axios.get('http://localhost:8501/api/status');
      return response.data;
    } catch (error: any) {
      return { error: error.message };
    }
  }
}));
