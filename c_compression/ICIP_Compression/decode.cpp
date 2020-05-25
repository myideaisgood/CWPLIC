#include "stdio.h"
#include "assert.h"
#include <windows.h>
#include <string>
#include <iostream>

#include "ppm_io.h"
#include "arithmetic_codec.h"
#include "encode.h"
#include "utils.h"

#pragma warning(disable: 4996)

unsigned int decodeMag_y(int ctx, Arithmetic_Codec *pCoder, Adaptive_Data_Model *pDm) {
	assert(SYMBOL_MAX == 16);

	unsigned int sym, left_over;
	int mag;

	sym = pCoder->decode(*pDm);

	if (sym < SYMBOL_MAX) {
		left_over = 0;
		mag = sym;
	}
	else if (sym == SYMBOL_MAX) {
		left_over = pCoder->get_bits(4);
		mag = 16 + left_over;
	}
	else if (sym == SYMBOL_MAX + 1) {
		left_over = pCoder->get_bits(5);
		mag = 32 + left_over;
	}
	else if (sym == SYMBOL_MAX + 2) {
		left_over = pCoder->get_bits(6);
		mag = 64 + left_over;
	}
	else {
		left_over = pCoder->get_bits(7);
		mag = 128 + left_over;
	}

	return mag;
}

unsigned int decodeMag_uv(int ctx, Arithmetic_Codec *pCoder, Adaptive_Data_Model *pDm) {
	assert(SYMBOL_MAX == 16);

	unsigned int sym, left_over;
	int mag;

	sym = pCoder->decode(*pDm);

	if (sym < SYMBOL_MAX) {
		left_over = 0;
		mag = sym;
	}
	else if (sym == SYMBOL_MAX) {
		left_over = pCoder->get_bits(4);
		mag = 16 + left_over;
	}
	else if (sym == SYMBOL_MAX + 1) {
		left_over = pCoder->get_bits(5);
		mag = 32 + left_over;
	}
	else if (sym == SYMBOL_MAX + 2) {
		left_over = pCoder->get_bits(6);
		mag = 64 + left_over;
	}
	else if (sym == SYMBOL_MAX + 3) {
		left_over = pCoder->get_bits(7);
		mag = 128 + left_over;
	}
	else {
		left_over = pCoder->get_bits(8);
		mag = 256 + left_over;
	}

	return mag;
}

int unmap_symbol_y(int sym, int left, float pred) {

	float Pd = pred + left;
	int P = UINT8(Pd);
	int X;
	bool flip = false;

	if (P >= 128) {
		Pd = 255 - Pd;
		P = 255 - P;
		flip = true;
	}

	if (Pd > P) {
		if (sym % 2 == 0) {
			X = P - (sym / 2);

			if (X > P || X < 0)
				X = sym;
		}
		else if (sym % 2 == 1) {
			X = P + ((sym + 1) / 2);

			if (X <= P || X > 2 * P)
				X = sym;			
		}
	}
	else {
		if (sym % 2 == 0) {
			X = P + (sym / 2);

			if (X < P || X >= 2*P)
				X = sym;
		}
		else if (sym % 2 == 1) {
			X = P - ((sym + 1) / 2);

			if (X >= P || X < 0)
				X = sym;
		}
	}

	if (flip) {
		X = 255 - X;
	}

	return X;

}

int unmap_symbol_uv(int sym, int left, float pred) {

	float Pd = pred + left;
	int P = int(Pd);
	int X;
	bool flip = false;

	if (P >= 256) {
		Pd = 511 - Pd;
		P = 511 - P;
		flip = true;
	}

	if (Pd > P) {
		if (sym % 2 == 0) {
			X = P - (sym / 2);

			if (X > P || X < 0)
				X = sym;
		}
		else if (sym % 2 == 1) {
			X = P + ((sym + 1) / 2);

			if (X <= P || X > 2 * P)
				X = sym;
		}
	}
	else {
		if (sym % 2 == 0) {
			X = P + (sym / 2);

			if (X < P || X >= 2 * P)
				X = sym;
		}
		else if (sym % 2 == 1) {
			X = P - ((sym + 1) / 2);

			if (X >= P || X < 0)
				X = sym;
		}
	}

	if (flip) {
		X = 511 - X;
	}

	return X;

}

void decode(FILE *fp_y, FILE *fp_u, FILE *fp_v, struct stNeuralNetwork *pNN_y, struct stNeuralNetwork *pNN_u, struct stNeuralNetwork *pNN_v, int **Y, int **U, int **V, int height, int width) {

	Arithmetic_Codec coder[3];
	Adaptive_Data_Model dm[3][NUM_CTX];

	int ctx_left = pNN_y->ctx_left;
	int ctx_up = pNN_y->ctx_up;
	int ctx_total = (ctx_left * 2 + 1)*ctx_up + ctx_left - 1;
	int hidden_unit = pNN_y->n_hidden;
	
	WEIGHT_TYPE nbr_y[3 * 11];

	int x, y;
	int numPix = 0;

	initModel_y(dm[0]);
	initModel_uv(dm[1]);
	initModel_uv(dm[2]);

	coder[0].set_buffer(width * height);
	coder[1].set_buffer(width * height);
	coder[2].set_buffer(width * height);

	coder[0].read_from_file(fp_y);
	coder[1].read_from_file(fp_u);
	coder[2].read_from_file(fp_v);

	coder[0].get_bits(16);
	coder[0].get_bits(16);

	for (y = 0; y < ctx_up; y++) {
		for (x = 0; x < width; x++) {
			Y[y][x] = coder[0].get_bits(8);
			U[y][x] = coder[1].get_bits(9);
			V[y][x] = coder[2].get_bits(9);
			numPix++;
		}
	}

	for (y = ctx_up; y < height; y++) {
		for (x = 0; x < pNN_y->ctx_left; x++) {
			Y[y][x] = coder[0].get_bits(8);
			U[y][x] = coder[1].get_bits(9);
			V[y][x] = coder[2].get_bits(9);
			numPix++;
		}

		for (x = ctx_left; x < width - ctx_left; x++) {

			int X_y;
			int X_u;
			int X_v;

			int left_y = Y[y][x - 1];
			int left_u = U[y][x - 1];
			int left_v = V[y][x - 1];

			int cnt = 0;

			for (int i = 0; i < ctx_up + 1; i++) {
				for (int j = 0; j < 2 * ctx_left + 1; j++) {

					if (i == ctx_up && j >= ctx_left - 1)
						break;

					nbr_y[cnt] = Y[y - (ctx_up - i)][x - (ctx_left - j)];
					nbr_y[cnt + ctx_total] = U[y - (ctx_up - i)][x - (ctx_left - j)];
					nbr_y[cnt + 2 * ctx_total] = V[y - (ctx_up - i)][x - (ctx_left - j)];

					cnt++;
				}
			}

			for (int i = 0; i < ctx_total; i++) {
				nbr_y[i] -= left_y;
				nbr_y[i + ctx_total] -= left_u;
				nbr_y[i + 2 * ctx_total] -= left_v;
			}

			float pred_y, ctx_y, pred_u, ctx_u, pred_v, ctx_v;
			float hidden_y[64], hidden_u[64], hidden_v[64];

			WEIGHT_TYPE input_u[64 + 33 + 2];
			WEIGHT_TYPE input_v[64 + 33 + 4];

			runNetwork(pNN_y, nbr_y, &pred_y, &ctx_y, hidden_y);

			int iCtx_y = calcContext(ctx_y, -1);
			unsigned int sym_y = decodeMag_y(iCtx_y, &coder[0], &dm[0][iCtx_y]);
			X_y = unmap_symbol_y(sym_y, left_y, pred_y);
			Y[y][x] = X_y;

			for (int i = 0; i < hidden_unit + 2 + 3 * ctx_total; i++) {
				if (i < hidden_unit)
					input_u[i] = hidden_y[i];
				else if (i < hidden_unit + 3 * ctx_total)
					input_u[i] = nbr_y[i - hidden_unit];
				else if (i == hidden_unit + 3 * ctx_total)
					input_u[i] = X_y - left_y;
				else
					input_u[i] = pred_y;
			}

			runNetwork(pNN_u, input_u, &pred_u, &ctx_u, hidden_u);
			
			int iCtx_u = calcContext(ctx_u, -1);
			unsigned int sym_u = decodeMag_uv(iCtx_u, &coder[1], &dm[1][iCtx_u]);
			X_u = unmap_symbol_uv(sym_u, left_u, pred_u);
			U[y][x] = X_u;

			for (int i = 0; i < hidden_unit + 4 + 3 * ctx_total; i++) {
				if (i < hidden_unit)
					input_v[i] = hidden_u[i];
				else if (i < hidden_unit + 3 * ctx_total)
					input_v[i] = nbr_y[i - hidden_unit];
				else if (i == hidden_unit + 3 * ctx_total)
					input_v[i] = X_y - left_y;
				else if (i == hidden_unit + 3 * ctx_total + 1)
					input_v[i] = pred_y;
				else if (i == hidden_unit + 3 * ctx_total + 2)
					input_v[i] = X_u - left_u;
				else
					input_v[i] = pred_u;
			}

			runNetwork(pNN_v, input_v, &pred_v, &ctx_v, hidden_v);

			int iCtx_v = calcContext(ctx_v, -1);
			unsigned int sym_v = decodeMag_uv(iCtx_v, &coder[2], &dm[2][iCtx_v]);
			X_v = unmap_symbol_uv(sym_v, left_v, pred_v);
			V[y][x] = X_v;

			numPix++;
		}

		for (x = width - ctx_left; x < width; x++) {
			Y[y][x] = coder[0].get_bits(8);
			U[y][x] = coder[1].get_bits(9);
			V[y][x] = coder[2].get_bits(9);
			numPix++;
		}
	}

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			U[y][x] -= 255;
			V[y][x] -= 255;
		}
	}

}

float runDecoder(char *codefile_y, char *codefile_u, char *codefile_v, char *outfile, char *weight_y, char *weight_u, char *weight_v) {

	FILE *fp_y, *fp_u, *fp_v;

	struct stNeuralNetwork NN_y, NN_u, NN_v;
	readWeight(weight_y, &NN_y);
	readWeight(weight_u, &NN_u);
	readWeight(weight_v, &NN_v);

	if ((fp_y = fopen(codefile_y, "rb")) == NULL) {
		fputs("Code file open error.\n", stderr);
		exit(1);
	}

	if ((fp_u = fopen(codefile_u, "rb")) == NULL) {
		fputs("Code file open error.\n", stderr);
		exit(1);
	}

	if ((fp_v = fopen(codefile_v, "rb")) == NULL) {
		fputs("Code file open error.\n", stderr);
		exit(1);
	}

	int width, height;

	Arithmetic_Codec coder;
	coder.set_buffer(3000*3000);
	coder.read_from_file(fp_y);
	width = coder.get_bits(16);
	height = coder.get_bits(16);

	fclose(fp_y);

	if ((fp_y = fopen(codefile_y, "rb")) == NULL) {
		fputs("Code file open error.\n", stderr);
		exit(1);
	}

	int **R, **G, **B;
	int **Y, **U, **V;
	Y = alloc2D(height, width);
	U = alloc2D(height, width);
	V = alloc2D(height, width);

	decode(fp_y, fp_u, fp_v, &NN_y, &NN_u, &NN_v, Y, U, V, height, width);

	fclose(fp_y);
	fclose(fp_u);
	fclose(fp_v);

	YUV2RGB(&Y, &U, &V, &R, &G, &B, &height, &width);

	writePPM(outfile, R, G, B, height, width, 255);

	free2D(R);
	free2D(G);
	free2D(B);
	free2D(Y);
	free2D(U);
	free2D(V);

	free2Dweight(NN_y.Win);
	free2Dweight(NN_y.Wout);
	free(NN_y.Bout);
	free(NN_y.B[0]);

	for (int i = 0; i < NN_y.n_layer - 2; i++) {
		free2Dweight(NN_y.W[i]);
		free(NN_y.B[i + 1]);
	}

	free2Dweight(NN_u.Win);
	free2Dweight(NN_u.Wout);
	free(NN_u.Bout);
	free(NN_u.B[0]);

	free2Dweight(NN_v.Win);
	free2Dweight(NN_v.Wout);
	free(NN_v.Bout);
	free(NN_v.B[0]);

	for (int i = 0; i < NN_u.n_layer - 2; i++) {
		free2Dweight(NN_u.W[i]);
		free(NN_u.B[i + 1]);
	}

	for (int i = 0; i < NN_v.n_layer - 2; i++) {
		free2Dweight(NN_v.W[i]);
		free(NN_v.B[i + 1]);
	}

	return 0;
}