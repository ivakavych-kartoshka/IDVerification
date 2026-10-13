'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { api, getToken, Verification } from '@/lib/api';
import LivenessChallenge from '@/components/LivenessChallenge';

type Side = 'front' | 'back';

export default function KycPage() {
  const router = useRouter();
  const [ready, setReady] = useState(false);

  const [front, setFront] = useState<File | null>(null);
  const [back, setBack] = useState<File | null>(null);
  const [livenessFrames, setLivenessFrames] = useState<File[]>([]);
  const [previews, setPreviews] = useState<Record<Side, string | undefined>>({
    front: undefined,
    back: undefined,
  });

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Verification | null>(null);

  useEffect(() => {
    if (!getToken()) {
      router.push('/login');
      return;
    }
    setReady(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function setPreview(side: Side, url?: string) {
    setPreviews((p) => {
      if (p[side]) URL.revokeObjectURL(p[side] as string);
      return { ...p, [side]: url };
    });
  }

  function handleFile(side: Side, file: File | null) {
    if (!file) return;
    if (side === 'front') setFront(file);
    if (side === 'back') setBack(file);
    setPreview(side, URL.createObjectURL(file));
  }

  function handleLiveness(frames: File[]) {
    setLivenessFrames(frames);
  }

  async function submit() {
    setError(null);
    if (!front) {
      setError('Cần ảnh CCCD mặt trước');
      return;
    }
    if (livenessFrames.length === 0) {
      setError('Cần hoàn thành kiểm tra người thật (liveness)');
      return;
    }
    setSubmitting(true);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append('front', front, front.name || 'front.jpg');
      if (back) fd.append('back', back, back.name || 'back.jpg');
      // The first "look straight" frame doubles as the selfie for face matching.
      fd.append('selfie', livenessFrames[0], livenessFrames[0].name || 'selfie.jpg');
      livenessFrames.forEach((f, i) =>
        fd.append('liveness', f, f.name || `liveness_${i}.jpg`),
      );
      const res = await api.submitVerification(fd);
      setResult(res);
    } catch (e: any) {
      setError(e.message || 'Gửi thất bại');
    } finally {
      setSubmitting(false);
    }
  }

  function resetAll() {
    setFront(null);
    setBack(null);
    setLivenessFrames([]);
    setPreview('front', undefined);
    setPreview('back', undefined);
    setResult(null);
    setError(null);
  }

  if (!ready) return <p className="sub">Đang kiểm tra đăng nhập...</p>;

  const steps = [
    { label: 'CCCD mặt trước', done: !!front },
    { label: 'CCCD mặt sau', done: !!back },
    { label: 'Kiểm tra người thật', done: livenessFrames.length > 0 },
    { label: 'Gửi xác thực', done: !!result },
  ];
  const activeStep = steps.findIndex((s) => !s.done);

  return (
    <>
      <h1>Xác thực CCCD (eKYC)</h1>
      <p className="sub">
        Upload 2 mặt CCCD, sau đó hoàn thành kiểm tra người thật (nhìn thẳng → quay trái → quay
        phải). Khung “nhìn thẳng” được dùng để so khớp khuôn mặt với ảnh CCCD.
      </p>

      <div className="card">
        {steps.map((s, i) => (
          <div
            key={s.label}
            className={'step' + (i === activeStep || (activeStep === -1 && i === steps.length - 1) ? ' active' : '')}
          >
            <span className="dot">{s.done ? '✓' : i + 1}</span>
            <span>{s.label}</span>
          </div>
        ))}
      </div>

      {error && <div className="alert error">{error}</div>}

      <div className="card">
        <div className="grid">
          <div>
            <h2>1. CCCD mặt trước *</h2>
            <div className="drop">
              <input
                type="file"
                accept="image/*"
                onChange={(e) => handleFile('front', e.target.files?.[0] ?? null)}
              />
              {previews.front && <img src={previews.front} alt="front" />}
            </div>
          </div>

          <div>
            <h2>2. CCCD mặt sau</h2>
            <div className="drop">
              <input
                type="file"
                accept="image/*"
                onChange={(e) => handleFile('back', e.target.files?.[0] ?? null)}
              />
              {previews.back && <img src={previews.back} alt="back" />}
            </div>
          </div>
        </div>

        <h2 style={{ marginTop: 24 }}>3. Kiểm tra người thật (liveness)</h2>
        <p className="sub">
          Nhìn thẳng, quay trái, quay phải theo hướng dẫn để chứng minh bạn là người thật.
        </p>
        <LivenessChallenge onComplete={handleLiveness} />

        <div className="row" style={{ marginTop: 24 }}>
          <button onClick={submit} disabled={submitting || !front} type="button">
            {submitting ? 'Đang xác thực...' : 'Gửi xác thực'}
          </button>
          <button className="secondary" onClick={resetAll} type="button">
            Làm lại
          </button>
        </div>
      </div>

      {result && (
        <div className="card">
          <h2>
            Kết quả: <span className={'badge ' + result.status}>{result.status}</span>
          </h2>
          <table>
            <tbody>
              <tr>
                <th>ID</th>
                <td>{result.id}</td>
              </tr>
              <tr>
                <th>Số CCCD (OCR)</th>
                <td>{result.documentNumber || '—'}</td>
              </tr>
              <tr>
                <th>Họ tên (QR)</th>
                <td>
                  {result.fullNameOnDoc ||
                    result.aiRawResult?.qr?.fields?.full_name ||
                    '—'}
                </td>
              </tr>
              <tr>
                <th>QR ↔ OCR</th>
                <td>
                  {result.aiRawResult?.qr_consistency?.status ??
                    (result.aiRawResult?.qr?.enabled === false ? 'TẮT' : '—')}
                  {result.aiRawResult?.qr_consistency?.score != null &&
                    ` (${result.aiRawResult.qr_consistency.score})`}
                </td>
              </tr>
              <tr>
                <th>Face match</th>
                <td>
                  {result.faceMatchResult?.status ?? '—'}{' '}
                  {result.faceMatchResult?.match_score != null &&
                    `(${result.faceMatchResult.match_score})`}
                </td>
              </tr>
              <tr>
                <th>Liveness (thụ động)</th>
                <td>{result.livenessResult?.passive?.status ?? '—'}</td>
              </tr>
              <tr>
                <th>Liveness (chủ động)</th>
                <td>
                  {result.livenessResult?.active?.status ?? '—'}
                  {result.livenessResult?.active?.message &&
                    result.livenessResult.active.status !== 'PASS' &&
                    ` — ${result.livenessResult.active.message}`}
                </td>
              </tr>
              <tr>
                <th>Lý do từ chối</th>
                <td>{result.rejectionReason || '—'}</td>
              </tr>
            </tbody>
          </table>
          <details style={{ marginTop: 12 }}>
            <summary style={{ cursor: 'pointer' }}>Chi tiết AI (raw)</summary>
            <pre>{JSON.stringify(result.aiRawResult ?? result, null, 2)}</pre>
          </details>
          <div className="row" style={{ marginTop: 16 }}>
            <button className="secondary" onClick={resetAll} type="button">
              Làm lại
            </button>
            <button className="secondary" onClick={() => router.push('/verifications')} type="button">
              Xem lịch sử
            </button>
          </div>
        </div>
      )}
    </>
  );
}
