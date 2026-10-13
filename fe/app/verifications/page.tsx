'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { api, fetchImageBlob, getToken, Verification } from '@/lib/api';

export default function VerificationsPage() {
  const router = useRouter();
  const [list, setList] = useState<Verification[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Verification | null>(null);
  const [images, setImages] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!getToken()) {
      router.push('/login');
      return;
    }
    api
      .listVerifications()
      .then(setList)
      .catch((e) => setError(e.message));
  }, [router]);

  async function open(v: Verification) {
    try {
      const full = await api.getVerification(v.id);
      setSelected(full);
      const imgs: Record<string, string> = {};
      for (const side of ['front', 'back', 'selfie'] as const) {
        try {
          imgs[side] = await fetchImageBlob(v.id, side);
        } catch {
          /* side not present */
        }
      }
      setImages(imgs);
    } catch (e: any) {
      setError(e.message);
    }
  }

  return (
    <>
      <h1>Lịch sử xác thực</h1>
      <p className="sub">Chỉ hiển thị verification của chính bạn (kiểm tra ownership ở BE).</p>
      {error && <div className="alert error">{error}</div>}

      <div className="card">
        {!list && <p className="sub">Đang tải...</p>}
        {list && list.length === 0 && <p className="sub">Chưa có verification nào.</p>}
        {list && list.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>Thời gian</th>
                <th>Loại</th>
                <th>Số CCCD</th>
                <th>Trạng thái</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {list.map((v) => (
                <tr key={v.id}>
                  <td>{new Date(v.createdAt).toLocaleString('vi-VN')}</td>
                  <td>{v.documentType}</td>
                  <td>{v.documentNumber || '—'}</td>
                  <td>
                    <span className={'badge ' + v.status}>{v.status}</span>
                  </td>
                  <td>
                    <button className="secondary" onClick={() => open(v)}>
                      Chi tiết
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {selected && (
        <div className="card">
          <h2>
            {selected.id} — <span className={'badge ' + selected.status}>{selected.status}</span>
          </h2>
          <div className="grid">
            {(['front', 'back', 'selfie'] as const).map((side) =>
              images[side] ? (
                <div key={side}>
                  <label>{side}</label>
                  <img src={images[side]} alt={side} style={{ maxWidth: '100%', borderRadius: 8 }} />
                </div>
              ) : null,
            )}
          </div>
          <details style={{ marginTop: 12 }} open>
            <summary style={{ cursor: 'pointer' }}>Thông tin QR</summary>
            {selected.qr ? (
              <table style={{ marginTop: 8 }}>
                <tbody>
                  <tr>
                    <th>Trạng thái QR ↔ OCR</th>
                    <td>
                      <span className={'badge ' + (selected.qr.consistencyStatus || '')}>
                        {selected.qr.consistencyStatus ?? '—'}
                      </span>{' '}
                      {selected.qr.consistencyScore != null &&
                        `(${selected.qr.consistencyScore})`}
                    </td>
                  </tr>
                  <tr>
                    <th>Đọc được QR</th>
                    <td>
                      {selected.qr.found ? `Có (${selected.qr.side ?? '—'}, ${selected.qr.reader ?? '—'})` : 'Không'}
                    </td>
                  </tr>
                  <tr>
                    <th>Số định danh (QR)</th>
                    <td>{selected.qr.documentNumber || '—'}</td>
                  </tr>
                  <tr>
                    <th>Số CMND cũ (QR)</th>
                    <td>{selected.qr.legacyId || '—'}</td>
                  </tr>
                  <tr>
                    <th>Họ tên (QR)</th>
                    <td>{selected.qr.fullName || '—'}</td>
                  </tr>
                  <tr>
                    <th>Ngày sinh (QR)</th>
                    <td>{selected.qr.dob || '—'}</td>
                  </tr>
                  <tr>
                    <th>Giới tính (QR)</th>
                    <td>{selected.qr.gender || '—'}</td>
                  </tr>
                  <tr>
                    <th>Nơi cư trú (QR)</th>
                    <td>{selected.qr.nativePlace || '—'}</td>
                  </tr>
                  <tr>
                    <th>Ngày cấp (QR)</th>
                    <td>{selected.qr.issueDate || '—'}</td>
                  </tr>
                  <tr>
                    <th>Trường khớp / lệch</th>
                    <td>
                      {(selected.qr.matchedFields ?? []).join(', ') || '—'} /{' '}
                      {(selected.qr.mismatchedFields ?? []).join(', ') || '—'}
                    </td>
                  </tr>
                  <tr>
                    <th>Payload QR</th>
                    <td style={{ wordBreak: 'break-all' }}>{selected.qr.payload || '—'}</td>
                  </tr>
                </tbody>
              </table>
            ) : (
              <p className="sub">Không có dữ liệu QR cho verification này.</p>
            )}
          </details>
          <details style={{ marginTop: 12 }}>
            <summary style={{ cursor: 'pointer' }}>Chi tiết AI</summary>
            <pre>{JSON.stringify(selected.aiRawResult ?? selected, null, 2)}</pre>
          </details>
        </div>
      )}
    </>
  );
}
