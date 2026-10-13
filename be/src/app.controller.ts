import { Controller, Get } from '@nestjs/common';
import { ApiTags } from '@nestjs/swagger';

@ApiTags('app')
@Controller()
export class AppController {
  @Get('health')
  health() {
    return { status: 'ok', service: 'cccd-verification-be' };
  }
}
