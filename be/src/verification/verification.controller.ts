import {
  Body,
  Controller,
  Get,
  Param,
  Post,
  Query,
  Req,
  Res,
  UploadedFiles,
  UseGuards,
  UseInterceptors,
} from '@nestjs/common';
import { FileFieldsInterceptor } from '@nestjs/platform-express';
import {
  ApiBearerAuth,
  ApiBody,
  ApiConsumes,
  ApiOperation,
  ApiTags,
} from '@nestjs/swagger';
import { Response } from 'express';
import { readFileSync } from 'fs';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { QueryVerificationDto } from './dto/query-verification.dto';
import { VerificationService, VerificationFiles } from './verification.service';

const ekycUpload = FileFieldsInterceptor(
  [
    { name: 'front', maxCount: 1 },
    { name: 'back', maxCount: 1 },
    { name: 'selfie', maxCount: 1 },
    { name: 'liveness', maxCount: 40 },
  ],
  { limits: { fileSize: 8 * 1024 * 1024 } },
);

const ekycBody = {
  schema: {
    type: 'object',
    properties: {
      front: { type: 'string', format: 'binary' },
      back: { type: 'string', format: 'binary' },
      selfie: { type: 'string', format: 'binary' },
      liveness: {
        type: 'array',
        items: { type: 'string', format: 'binary' },
        description: 'Ordered challenge frames (look straight / turn left / turn right)',
      },
      documentType: { type: 'string', example: 'CCCD' },
    },
    required: ['front'],
  },
};

@ApiTags('verifications')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('verifications')
export class VerificationController {
  constructor(private readonly service: VerificationService) {}

  @Post()
  @ApiOperation({ summary: 'Submit CCCD front + back + selfie and run AI verification' })
  @ApiConsumes('multipart/form-data')
  @ApiBody(ekycBody)
  @UseInterceptors(ekycUpload)
  create(@Req() req: any, @UploadedFiles() files: VerificationFiles, @Body() body: any) {
    return this.service.createAndProcess(req.user.id, files, body?.documentType || 'CCCD');
  }

  @Get()
  @ApiOperation({ summary: 'List my verifications' })
  list(@Req() req: any, @Query() query: QueryVerificationDto) {
    return this.service.findAllForUser(req.user.id, query.status, query.documentNumber);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get one of my verifications' })
  getOne(@Req() req: any, @Param('id') id: string) {
    return this.service.findOneForUser(req.user.id, id);
  }

  @Get(':id/image/:side')
  @ApiOperation({ summary: 'Stream a stored image (front | back | selfie) with ownership check' })
  async image(
    @Req() req: any,
    @Param('id') id: string,
    @Param('side') side: 'front' | 'back' | 'selfie',
    @Res() res: Response,
  ) {
    const meta = await this.service.getPreviewMeta(req.user.id, id, side);
    const buffer = readFileSync(meta.absPath);
    const ext = meta.name.split('.').pop()?.toLowerCase();
    const mime =
      ext === 'png' ? 'image/png' : ext === 'webp' ? 'image/webp' : 'image/jpeg';
    res.setHeader('Content-Type', mime);
    res.setHeader('Cache-Control', 'private, no-store');
    res.send(buffer);
  }

  @Post(':id/retry')
  @ApiOperation({ summary: 'Retry verification (optionally replacing images)' })
  @ApiConsumes('multipart/form-data')
  @ApiBody({ schema: ekycBody.schema })
  @UseInterceptors(ekycUpload)
  retry(@Req() req: any, @Param('id') id: string, @UploadedFiles() files: VerificationFiles) {
    return this.service.retry(req.user.id, id, files);
  }
}
