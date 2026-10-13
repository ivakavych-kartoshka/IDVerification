import { BadRequestException, Injectable, NotFoundException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { randomUUID } from 'crypto';
import { createReadStream, existsSync, mkdirSync, promises as fs } from 'fs';
import { extname, join, resolve } from 'path';

const ALLOWED_MIME = new Set(['image/jpeg', 'image/jpg', 'image/png', 'image/webp']);
const MAX_BYTES = 8 * 1024 * 1024;

const EXT_BY_MIME: Record<string, string> = {
  'image/jpeg': '.jpg',
  'image/jpg': '.jpg',
  'image/png': '.png',
  'image/webp': '.webp',
};

@Injectable()
export class StorageService {
  private readonly rootDir: string;

  constructor(config: ConfigService) {
    const dir = config.get<string>('UPLOAD_DIR') || 'uploads';
    this.rootDir = resolve(process.cwd(), dir);
    if (!existsSync(this.rootDir)) {
      mkdirSync(this.rootDir, { recursive: true });
    }
  }

  getUserDir(userId: string): string {
    const dir = join(this.rootDir, userId);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }
    return dir;
  }

  async save(userId: string, file: Express.Multer.File): Promise<string> {
    if (!file || !file.buffer) {
      throw new BadRequestException('Empty file');
    }
    if (file.size > MAX_BYTES) {
      throw new BadRequestException('File too large (max 8MB)');
    }
    if (!ALLOWED_MIME.has(file.mimetype)) {
      throw new BadRequestException(`Unsupported file type: ${file.mimetype}`);
    }
    const ext =
      EXT_BY_MIME[file.mimetype] || (extname(file.originalname) || '.jpg').toLowerCase();
    const name = `${randomUUID()}${ext}`;
    const abs = join(this.getUserDir(userId), name);
    await fs.writeFile(abs, file.buffer);
    return name;
  }

  resolve(userId: string, name: string): string {
    const abs = join(this.rootDir, userId, name);
    if (!existsSync(abs)) {
      throw new NotFoundException('Stored file not found');
    }
    return abs;
  }

  createStream(userId: string, name: string) {
    return createReadStream(this.resolve(userId, name));
  }

  async remove(userId: string, name?: string | null): Promise<void> {
    if (!name) return;
    try {
      await fs.unlink(join(this.rootDir, userId, name));
    } catch {
      /* ignore */
    }
  }
}
