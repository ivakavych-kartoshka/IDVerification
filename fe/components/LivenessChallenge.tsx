'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/lib/api';

const STRAIGHT_MAX = 12;
const TURN_MIN = 18;
const POLL_MS = 550;

const PHASES = [
  { title: 'Nhìn thẳng', hint: 'Giữ mặt nhìn thẳng vào camera' },
  { title: 'Quay sang trái', hint: 'Từ từ quay đầu sang trái rồi giữ ~1 giây' },
  { title: 'Quay sang phải', hint: 'Quay đầu sang phải rồi giữ ~1 giây' },
];

export default function LivenessChallenge({
  onComplete,
}: {
  onComplete: (frames: File[]) => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const phaseRef = useRef(0);
  const dirRef = useRef<number>(0);
  const framesRef = useRef<File[]>([]);
  const busyRef = useRef(false);
  const doneRef = useRef(false);

  const [running, setRunning] = useState(false);
  const [phaseIdx, setPhaseIdx] = useState(0);
  const [yaw, setYaw] = useState<number | null>(null);
  const [count, setCount] = useState(0);
  const [status, setStatus] = useState('');
  const [error, setError] = useState<string | null>(null);

  const stop = useCallback(() => {
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setRunning(false);
  }, []);

  useEffect(() => () => stop(), [stop]);

  function grabFrame(): Promise<File | null> {
    return new Promise((resolve) => {
      const video = videoRef.current;
      if (!video || !video.videoWidth) return resolve(null);
      const canvas = document.createElement('canvas');
      canvas.width = 320;
      canvas.height = 240;
      const ctx = canvas.getContext('2d');
      if (!ctx) return resolve(null);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (blob) => {
          if (!blob) return resolve(null);
          const name = `liveness_${Date.now()}_${Math.random().toString(36).slice(2, 7)}.jpg`;
          resolve(new File([blob], name, { type: 'image/jpeg' }));
        },
        'image/jpeg',
        0.7,
      );
    });
  }

  async function collect(keep: number) {
    for (let i = 0; i < keep; i++) {
      const f = await grabFrame();
      if (f) framesRef.current.push(f);
      setCount(framesRef.current.length);
      if (i < keep - 1) await new Promise((r) => setTimeout(r, 120));
    }
  }

  async function tick() {
    if (busyRef.current || doneRef.current) return;
    busyRef.current = true;
    try {
      const frame = await grabFrame();
      if (!frame) return;
      const pose = await api.livenessPose(frame, frame.name);
      if (!pose.face_found || pose.yaw == null) {
        setYaw(null);
        setStatus('Không thấy khuôn mặt — hãy đưa mặt vào khung hình');
        return;
      }
      setYaw(pose.yaw);
      const phase = phaseRef.current;

      if (phase === 0) {
        if (Math.abs(pose.yaw) <= STRAIGHT_MAX) {
          await collect(2);
          phaseRef.current = 1;
          setPhaseIdx(1);
          setStatus('Tốt! Giờ quay đầu sang trái');
        } else {
          setStatus('Hãy nhìn thẳng vào camera');
        }
      } else if (phase === 1) {
        if (Math.abs(pose.yaw) >= TURN_MIN) {
          dirRef.current = pose.yaw > 0 ? 1 : -1;
          await collect(2);
          phaseRef.current = 2;
          setPhaseIdx(2);
          setStatus('Tốt! Giờ quay đầu sang phải');
        } else {
          setStatus('Quay đầu sang trái và giữ ~1 giây');
        }
      } else if (phase === 2) {
        const sameDirection =
          dirRef.current > 0 ? pose.yaw >= TURN_MIN : pose.yaw <= -TURN_MIN;
        if (!sameDirection && Math.abs(pose.yaw) >= TURN_MIN) {
          await collect(2);
          doneRef.current = true;
          setPhaseIdx(3);
          setStatus('Hoàn tất kiểm tra người thật');
          stop();
          onComplete(framesRef.current);
        } else {
          setStatus('Quay đầu sang phải và giữ ~1 giây');
        }
      }
    } catch (e: any) {
      setStatus('Lỗi kiểm tra: ' + (e?.message || 'không xác định'));
    } finally {
      busyRef.current = false;
    }
  }

  async function start() {
    setError(null);
    framesRef.current = [];
    phaseRef.current = 0;
    dirRef.current = 0;
    doneRef.current = false;
    busyRef.current = false;
    setPhaseIdx(0);
    setYaw(null);
    setCount(0);
    setStatus('Đang mở camera...');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 },
        audio: false,
      });
      streamRef.current = stream;
      setRunning(true);
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => {});
        }
      }, 50);
      timerRef.current = window.setInterval(tick, POLL_MS);
    } catch (e: any) {
      setError('Không mở được camera: ' + (e?.message || e?.name));
    }
  }

  const done = phaseIdx >= PHASES.length;

  return (
    <div>
      <div className="row">
        {!running && !done && (
          <button className="secondary" onClick={start} type="button">
            Bắt đầu kiểm tra người thật
          </button>
        )}
        {running && (
          <button className="secondary" onClick={stop} type="button">
            Dừng
          </button>
        )}
        {done && (
          <button className="secondary" onClick={start} type="button">
            Chụp lại
          </button>
        )}
        {count > 0 && <span className="sub">Đã thu {count} khung hình</span>}
      </div>

      {error && <div className="alert error">{error}</div>}

      <div className="row" style={{ marginTop: 12, alignItems: 'flex-start' }}>
        <div style={{ display: running ? 'block' : 'none' }}>
          <video
            ref={videoRef}
            playsInline
            muted
            style={{ transform: 'scaleX(-1)', borderRadius: 8, maxWidth: 320 }}
          />
        </div>
        {(running || done) && (
          <div style={{ minWidth: 200 }}>
            <ol style={{ margin: 0, paddingLeft: 18 }}>
              {PHASES.map((p, i) => (
                <li
                  key={p.title}
                  style={{
                    fontWeight: i === phaseIdx ? 700 : 400,
                    opacity: i > phaseIdx ? 0.5 : 1,
                  }}
                >
                  {p.title}
                  {i < phaseIdx ? ' ✓' : ''}
                </li>
              ))}
            </ol>
            <p className="sub" style={{ marginTop: 8 }}>{status}</p>
            {yaw != null && <p className="sub">Góc quay: {yaw}°</p>}
          </div>
        )}
      </div>
    </div>
  );
}
