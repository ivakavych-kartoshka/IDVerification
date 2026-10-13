export const API_URL =
  process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000/api';

const TOKEN_KEY = 'cccd_token';

export function getToken(): string | null {
  if (typeof window === 'undefined') return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string) {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
  if (typeof window === 'undefined') return;
  window.localStorage.removeItem(TOKEN_KEY);
}

export interface User {
  id: string;
  email: string;
  fullName?: string | null;
  role?: string;
}

export interface Verification {
  id: string;
  documentType: string;
  documentNumber?: string | null;
  fullNameOnDoc?: string | null;
  status: 'PENDING' | 'PROCESSING' | 'VERIFIED' | 'REJECTED' | 'ERROR';
  rejectionReason?: string | null;
  ocrResult?: any;
  documentQuality?: any;
  faceMatchResult?: any;
  livenessResult?: any;
  aiRawResult?: any;
  qr?: VerificationQr | null;
  createdAt: string;
  updatedAt: string;
}

export interface VerificationQr {
  id: string;
  found: boolean;
  reader?: string | null;
  side?: string | null;
  payload?: string | null;
  documentNumber?: string | null;
  legacyId?: string | null;
  fullName?: string | null;
  dob?: string | null;
  gender?: string | null;
  nativePlace?: string | null;
  issueDate?: string | null;
  consistencyStatus?: string | null;
  consistencyScore?: number | null;
  matchedFields?: string[];
  mismatchedFields?: string[];
  createdAt: string;
  updatedAt: string;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  if (options.body && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }

  const res = await fetch(`${API_URL}${path}`, { ...options, headers });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;

  if (!res.ok) {
    const msg =
      (data && (data.message || data.error)) ||
      `Request failed with status ${res.status}`;
    throw new ApiError(res.status, Array.isArray(msg) ? msg.join(', ') : msg);
  }
  return data as T;
}

export const api = {
  register(body: { email: string; password: string; fullName?: string }) {
    return request<{ user: User; accessToken: string }>('/auth/register', {
      method: 'POST',
      body: JSON.stringify(body),
    });
  },
  login(body: { email: string; password: string }) {
    return request<{ user: User; accessToken: string }>('/auth/login', {
      method: 'POST',
      body: JSON.stringify(body),
    });
  },
  me() {
    return request<User>('/auth/me');
  },
  listVerifications() {
    return request<Verification[]>('/verifications');
  },
  getVerification(id: string) {
    return request<Verification>(`/verifications/${id}`);
  },
  submitVerification(form: FormData) {
    return request<Verification>('/verifications', { method: 'POST', body: form });
  },
  retryVerification(id: string, form: FormData) {
    return request<Verification>(`/verifications/${id}/retry`, {
      method: 'POST',
      body: form,
    });
  },
  livenessPose(file: Blob, filename = 'frame.jpg') {
    const fd = new FormData();
    fd.append('file', file, filename);
    return request<{
      face_found: boolean;
      yaw: number | null;
      pitch: number | null;
      roll: number | null;
    }>('/ai/liveness/pose', { method: 'POST', body: fd });
  },
};

export function imageUrl(id: string, side: 'front' | 'back' | 'selfie') {
  return `${API_URL}/verifications/${id}/image/${side}`;
}

export async function fetchImageBlob(
  id: string,
  side: 'front' | 'back' | 'selfie',
): Promise<string> {
  const token = getToken();
  const res = await fetch(imageUrl(id, side), {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!res.ok) throw new ApiError(res.status, 'Cannot load image');
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}
