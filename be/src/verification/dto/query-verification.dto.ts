import { ApiPropertyOptional } from '@nestjs/swagger';
import { IsEnum, IsOptional, IsString } from 'class-validator';

export enum ImageSide {
  front = 'front',
  back = 'back',
  selfie = 'selfie',
}

export class QueryVerificationDto {
  @ApiPropertyOptional({ enum: ['PENDING', 'PROCESSING', 'VERIFIED', 'REJECTED', 'ERROR'] })
  @IsOptional()
  @IsEnum(['PENDING', 'PROCESSING', 'VERIFIED', 'REJECTED', 'ERROR'])
  status?: string;

  @ApiPropertyOptional()
  @IsOptional()
  @IsString()
  documentNumber?: string;
}

export class ImageSideParam {
  @IsEnum(ImageSide)
  side!: ImageSide;
}
