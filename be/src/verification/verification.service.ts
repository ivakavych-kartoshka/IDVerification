import {
  BadRequestException,
  ForbiddenException,
  Injectable,
  Logger,
  NotFoundException,
} from '@nestjs/common';
import { readFileSync } from 'fs';
import { VerificationStatus } from '@prisma/client';
import { AiService, AiFile } from '../ai/ai.service';
import { PrismaService } from '../prisma/prisma.service';
import { StorageService } from '../storage/storage.service';

export interface VerificationFiles {
  front?: Express.Multer.File[];
  back?: Express.Multer.File[];
  selfie?: Express.Multer.File[];
  liveness?: Express.Multer.File[];
}

@Injectable()
export class VerificationService {
  private readonly logger = new Logger(VerificationService.name);

  constructor(
    private readonly prisma: PrismaService,
    private readonly storage: StorageService,
    private readonly ai: AiService,
  ) {}

  private toAiFile(file: Express.Multer.File): AiFile {
    return { buffer: file.buffer, filename: file.originalname, mimetype: file.mimetype };
  }

  async createAndProcess(userId: string, files: VerificationFiles, documentType = 'CCCD') {
    const front = files.front?.[0];
    if (!front) {
      throw new BadRequestException('front image is required');
    }
    const back = files.back?.[0];
    const selfie = files.selfie?.[0] ?? files.liveness?.[0];

    const frontName = await this.storage.save(userId, front);
    const backName = back ? await this.storage.save(userId, back) : null;
    const selfieName = selfie ? await this.storage.save(userId, selfie) : null;

    const record = await this.prisma.identityVerification.create({
      data: {
        userId,
        documentType,
        frontImagePath: frontName,
        backImagePath: backName,
        selfiePath: selfieName,
        status: VerificationStatus.PENDING,
      },
    });

    return this.process(record.id, { front, back, selfie, liveness: files.liveness });
  }

  async process(
    id: string,
    files: {
      front: Express.Multer.File;
      back?: Express.Multer.File;
      selfie?: Express.Multer.File;
      liveness?: Express.Multer.File[];
    },
  ) {
    await this.prisma.identityVerification.update({
      where: { id },
      data: { status: VerificationStatus.PROCESSING },
    });

    try {
      const result = await this.ai.verify(
        this.toAiFile(files.front),
        files.back ? this.toAiFile(files.back) : undefined,
        files.selfie ? this.toAiFile(files.selfie) : undefined,
        files.liveness?.length ? files.liveness.map((f) => this.toAiFile(f)) : undefined,
      );

      const status = this.mapStatus(result.status);
      const documentNumber =
        result?.ocr?.front?.document_number ?? result?.qr?.fields?.document_number ?? null;
      const fullNameOnDoc = result?.qr?.fields?.full_name ?? null;

      const duplicate = documentNumber
        ? await this.findDuplicate(documentNumber, id)
        : null;

      if (duplicate) {
        const saved = await this.prisma.identityVerification.update({
          where: { id },
          data: {
            status: VerificationStatus.REJECTED,
            documentNumber,
            fullNameOnDoc,
            ocrResult: result.ocr ?? undefined,
            documentQuality: result.document_quality ?? undefined,
            faceMatchResult: (result.face_match ?? result.face) ?? undefined,
            livenessResult: {
              passive: result.passive_liveness ?? null,
              active: result.active_liveness ?? null,
            },
            aiRawResult: result as any,
            rejectionReason: `CCCD ${documentNumber} is already verified`,
          },
        });
        await this.saveQr(id, result);
        return saved;
      }

      const updated = await this.prisma.identityVerification.update({
        where: { id },
        data: {
          status,
          documentNumber,
          fullNameOnDoc,
          ocrResult: result.ocr ?? undefined,
          documentQuality: result.document_quality ?? undefined,
          faceMatchResult: (result.face_match ?? result.face) ?? undefined,
          livenessResult: {
            passive: result.passive_liveness ?? null,
            active: result.active_liveness ?? null,
          },
          aiRawResult: result as any,
          rejectionReason: result.rejection_reason ?? null,
        },
      });
      await this.saveQr(id, result);
      return updated;
    } catch (error: any) {
      this.logger.error(`Verification ${id} failed: ${error.message}`);
      await this.prisma.identityVerification.update({
        where: { id },
        data: {
          status: VerificationStatus.ERROR,
          rejectionReason: `AI error: ${error.message}`,
        },
      });
      throw error;
    }
  }

  private async findDuplicate(
    documentNumber: string,
    excludeVerificationId: string,
  ) {
    return this.prisma.identityVerification.findFirst({
      where: {
        documentNumber,
        status: VerificationStatus.VERIFIED,
        id: { not: excludeVerificationId },
      },
      select: { id: true, userId: true },
    });
  }

  private async saveQr(verificationId: string, result: any) {
    const qr = result?.qr;
    if (!qr) return;

    const front = qr.front ?? {};
    const back = qr.back ?? {};
    const used = back.found ? back : front.found ? front : null;
    const fields = qr.fields ?? {};
    const consistency = result?.qr_consistency ?? {};

    const data = {
      found: !!used,
      reader: used?.reader ?? null,
      side: back.found ? 'back' : front.found ? 'front' : null,
      payload: used?.payload ?? null,
      documentNumber: fields.document_number ?? null,
      legacyId: fields.legacy_id ?? null,
      fullName: fields.full_name ?? null,
      dob: fields.dob ?? null,
      gender: fields.gender ?? null,
      nativePlace: fields.native_place ?? null,
      issueDate: fields.issue_date ?? null,
      consistencyStatus: consistency.status ?? null,
      consistencyScore: consistency.score ?? null,
      matchedFields: consistency.matched_fields ?? [],
      mismatchedFields: consistency.mismatched_fields ?? [],
      raw: qr as any,
    };

    await this.prisma.verificationQr.upsert({
      where: { verificationId },
      create: { verificationId, ...data },
      update: data,
    });
  }

  private mapStatus(aiStatus?: string): VerificationStatus {
    switch ((aiStatus || '').toUpperCase()) {
      case 'VERIFIED':
        return VerificationStatus.VERIFIED;
      case 'REJECTED':
        return VerificationStatus.REJECTED;
      default:
        return VerificationStatus.PROCESSING;
    }
  }

  async findAllForUser(userId: string, status?: string, documentNumber?: string) {
    return this.prisma.identityVerification.findMany({
      where: {
        userId,
        ...(status ? { status: status as VerificationStatus } : {}),
        ...(documentNumber ? { documentNumber } : {}),
      },
      orderBy: { createdAt: 'desc' },
      select: {
        id: true,
        documentType: true,
        documentNumber: true,
        status: true,
        rejectionReason: true,
        createdAt: true,
        updatedAt: true,
      },
    });
  }

  async findOneForUser(userId: string, id: string) {
    const record = await this.prisma.identityVerification.findUnique({
      where: { id },
      include: { qr: true },
    });
    if (!record) {
      throw new NotFoundException('Verification not found');
    }
    if (record.userId !== userId) {
      throw new ForbiddenException('You cannot access this verification');
    }
    return record;
  }

  getPreviewMeta(userId: string, id: string, side: 'front' | 'back' | 'selfie') {
    return this.findOneForUser(userId, id).then((r) => {
      const name =
        side === 'front' ? r.frontImagePath : side === 'back' ? r.backImagePath : r.selfiePath;
      if (!name) {
        throw new NotFoundException(`No ${side} image for this verification`);
      }
      return { name, absPath: this.storage.resolve(userId, name) };
    });
  }

  async retry(userId: string, id: string, files: VerificationFiles) {
    const record = await this.prisma.identityVerification.findUnique({ where: { id } });
    if (!record) {
      throw new NotFoundException('Verification not found');
    }
    if (record.userId !== userId) {
      throw new ForbiddenException('You cannot access this verification');
    }

    if (files.front?.[0]) {
      const name = await this.storage.save(userId, files.front[0]);
      await this.prisma.identityVerification.update({
        where: { id },
        data: { frontImagePath: name },
      });
    }
    if (files.back?.[0]) {
      const name = await this.storage.save(userId, files.back[0]);
      await this.prisma.identityVerification.update({
        where: { id },
        data: { backImagePath: name },
      });
    }
    if (files.selfie?.[0]) {
      const name = await this.storage.save(userId, files.selfie[0]);
      await this.prisma.identityVerification.update({
        where: { id },
        data: { selfiePath: name },
      });
    }

    const loaded = await this.prisma.identityVerification.findUnique({ where: { id } });
    if (!loaded) throw new NotFoundException('Verification not found');

    return this.process(id, {
      front: files.front?.[0] ?? this.loadAsMulter(userId, loaded.frontImagePath, 'front.jpg'),
      back: files.back?.[0] ?? this.optionalMulter(userId, loaded.backImagePath, 'back.jpg'),
      selfie: files.selfie?.[0] ?? this.optionalMulter(userId, loaded.selfiePath, 'selfie.jpg'),
      liveness: files.liveness,
    });
  }

  private loadAsMulter(userId: string, name: string, fallbackName: string): Express.Multer.File {
    const abs = this.storage.resolve(userId, name);
    const buffer = readFileSync(abs);
    return { buffer, originalname: fallbackName, mimetype: 'image/jpeg', size: buffer.length } as unknown as Express.Multer.File;
  }

  private optionalMulter(
    userId: string,
    name: string | null,
    fallbackName: string,
  ): Express.Multer.File | undefined {
    if (!name) return undefined;
    return this.loadAsMulter(userId, name, fallbackName);
  }
}
