import { readFileSync } from 'node:fs';
import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

const app = JSON.parse(readFileSync(new URL('../../mlx-bun/package.json', import.meta.url), 'utf8'));

export const collections = {
	docs: defineCollection({
		loader: docsLoader(),
		schema: docsSchema({
			extend: z.object({
				banner: z.object({ content: z.string() }).default({
					content: `mlx-bun ${app.version} is beta and actively developed. APIs may change; breaking changes will be called out in release notes.`,
				}),
			}),
		}),
	}),
};
