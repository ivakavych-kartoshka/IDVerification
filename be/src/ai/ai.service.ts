import {
  GatewayTimeoutException,
  Injectable,
  ServiceUnavailableException,
} from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import axios, { AxiosInstance } from 'axios';
import FormData from 'form-data';

export interface AiFile {
  buffer: Buffer;
  filename: string;
  mimetype: string;
}

export interface AiVerifyResult {
  status: string;
  ocr?: any;
  qr?: any;
  qr_consistency?: any;
  document_quality?: any;
  face?: any;
  face_match?: any;
  passive_liveness?: any;
  active_liveness?: any;
  rejection_reason?: string | null;
}

@Injectable()
export class AiService {
  private readonly http: AxiosInstance;
  private readonly baseUrl: string;
  private readonly mock: boolean;

  constructor(config: ConfigService) {
    this.baseUrl = config.get<string>('AI_SERVICE_URL') || 'http://127.0.0.1:8001';
    this.mock = (config.get<string>('AI_MOCK') || 'false').toLowerCase() === 'true';
    const timeout = Number(config.get<string>('AI_SERVICE_TIMEOUT_MS') || 60000);
    this.http = axios.create({ baseURL: this.baseUrl, timeout });
  }

  async health() {
    try {
      const { data } = await this.http.get('/healthz');
      return { reachable: true, ...data };
    } catch (e: any) {
      return { reachable: false, error: e.message };
    }
  }

  async verify(
    front: AiFile,
    back?: AiFile,
    selfie?: AiFile,
    liveness?: AiFile[],
  ): Promise<AiVerifyResult> {
    if (this.mock) {
      return this.mockResult();
    }
    try {
      const form = this.buildForm(front, back, selfie, liveness);
      const { data } = await this.http.post('/verify/', form, {
        headers: form.getHeaders(),
        maxBodyLength: Infinity,
        maxContentLength: Infinity,
      });
      return data?.result ?? data;
    } catch (e: any) {
      if (e.code === 'ECONNABORTED' || e.code === 'ETIMEDOUT') {
        throw new GatewayTimeoutException('AI service timed out');
      }
      if (e.response) {
        throw new ServiceUnavailableException(
          `AI service error (${e.response.status}): ${JSON.stringify(e.response.data)}`,
        );
      }
      throw new ServiceUnavailableException(`AI service unavailable: ${e.message}`);
    }
  }

  async ocr(file: AiFile) {
    try {
      const form = new FormData();
      form.append('file', file.buffer, {
        filename: file.filename,
        contentType: file.mimetype,
      });
      const { data } = await this.http.post('/ocr/extract', form, {
        headers: form.getHeaders(),
        maxBodyLength: Infinity,
      });
      return data?.result ?? data;
    } catch (e: any) {
      if (e.code === 'ECONNABORTED' || e.code === 'ETIMEDOUT') {
        throw new GatewayTimeoutException('AI service timed out');
      }
      throw new ServiceUnavailableException(`AI service unavailable: ${e.message}`);
    }
  }

  async livenessPose(file: AiFile) {
    try {
      const form = new FormData();
      form.append('file', file.buffer, {
        filename: file.filename,
        contentType: file.mimetype,
      });
      const { data } = await this.http.post('/liveness/pose', form, {
        headers: form.getHeaders(),
        maxBodyLength: Infinity,
      });
      return data?.result ?? data;
    } catch (e: any) {
      if (e.code === 'ECONNABORTED' || e.code === 'ETIMEDOUT') {
        throw new GatewayTimeoutException('AI service timed out');
      }
      throw new ServiceUnavailableException(`AI service unavailable: ${e.message}`);
    }
  }

  private buildForm(
    front: AiFile,
    back?: AiFile,
    selfie?: AiFile,
    liveness?: AiFile[],
  ): FormData {
    const form = new FormData();
    form.append('front', front.buffer, {
      filename: front.filename,
      contentType: front.mimetype,
    });
    if (back) {
      form.append('back', back.buffer, {
        filename: back.filename,
        contentType: back.mimetype,
      });
    }
    if (selfie) {
      form.append('selfie', selfie.buffer, {
        filename: selfie.filename,
        contentType: selfie.mimetype,
      });
    }
    for (const frame of liveness ?? []) {
      form.append('liveness', frame.buffer, {
        filename: frame.filename,
        contentType: frame.mimetype,
      });
    }
    return form;
  }

  private mockResult(): AiVerifyResult {
    return {
      status: 'VERIFIED',
      ocr: {
        front: {
          lines: [{ text: 'CONG HOA XA HOI CHU NGHIA VIET NAM', score: 0.99 }],
          full_text: 'CONG HOA XA HOI CHU NGHIA VIET NAM 012345678901',
          document_number: '012345678901',
        },
        back: { lines: [], full_text: '', document_number: null },
      },
      document_quality: { front_blur_var: 120.5, back_blur_var: 98.2 },
      qr: {
        front: { found: false, payload: null, fields: null, reader: null },
        back: {
          found: true,
          payload:
            '012345678901|251043755|NGUYEN VAN A|01011990|Nam|Ha Noi|01012021',
          fields: {
            document_number: '012345678901',
            legacy_id: '251043755',
            full_name: 'NGUYEN VAN A',
            dob: '01011990',
            gender: 'Nam',
            native_place: 'Ha Noi',
            issue_date: '01012021',
          },
          reader: 'opencv',
        },
        fields: {
          document_number: '012345678901',
          legacy_id: '251043755',
          full_name: 'NGUYEN VAN A',
          dob: '01011990',
          gender: 'Nam',
          native_place: 'Ha Noi',
          issue_date: '01012021',
        },
        enabled: true,
      },
      qr_consistency: {
        status: 'MATCH',
        score: 1.0,
        checked_fields: ['document_number', 'full_name', 'dob', 'gender'],
        matched_fields: ['document_number', 'full_name', 'dob', 'gender'],
        mismatched_fields: [],
        not_checked_fields: [],
        details: {},
      },
      face: { face_count_selfie: 1, face_count_id: 1 },
      face_match: { match_score: 0.92, status: 'MATCH' },
      passive_liveness: { is_live: true, blur_var: 130.1, status: 'PASS' },
      active_liveness: { status: 'NOT_REQUIRED' },
      rejection_reason: null,
    };
  }
}
