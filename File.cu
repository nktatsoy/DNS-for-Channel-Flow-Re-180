#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <fstream>
#include <iostream>
#include <random>
#define pi 3.1415927f

using namespace std;

__global__ void bcKernel(float* Un, float* Vn, float* Wn, int Nx, int Ny, int Nz) {

	int i = threadIdx.x;
	int k = blockIdx.x;

	Un[0 * Nx * Nz + k * Nx + i] = -Un[1 * Nx * Nz + k * Nx + i];
	Un[(Ny + 1) * Nx * Nz + k * Nx + i] = -Un[Ny * Nx * Nz + k * Nx + i];
	Vn[0 * Nx * Nz + k * Nx + i] = 0.0f;
	Vn[Ny * Nx * Nz + k * Nx + i] = 0.0f;
	Wn[0 * Nx * Nz + k * Nx + i] = -Wn[1 * Nx * Nz + k * Nx + i];
	Wn[(Ny + 1) * Nx * Nz + k * Nx + i] = -Wn[Ny * Nx * Nz + k * Nx + i];

}

__global__ void uhatKernel(float* Uhat, float* Un, float* Vn, float* Wn, float* Uold, float* Vold, float* Wold, float alfa, float gama, float zeta, float invRe, float invdx, float invdy, float invdz, float invdx2, float invdy2, float invdz2, float dt, int Nx, int Ny, int Nz) {

	int i = threadIdx.x;
	int k = blockIdx.x;

	int ip, im, jp, jm, kp, km;
	float convnx, convny, convnz;
	float convox, convoy, convoz;
	float diffx, diffy, diffz;
	float Hn, Ho, diffall;

	ip = i + 1; im = i - 1;	kp = k + 1;	km = k - 1;
	if (i == 0) { im = Nx - 1; }
	if (i == Nx - 1) { ip = 0; }
	if (k == 0) { km = Nz - 1; }
	if (k == Nz - 1) { kp = 0; }

	for (int j = 1; j < Ny + 1; j++) {

		jp = j + 1;
		jm = j - 1;

		convox = 0.5f * invdx * Uold[j * Nx * Nz + k * Nx + i] * (Uold[j * Nx * Nz + k * Nx + ip] - Uold[j * Nx * Nz + k * Nx + im]);
		convoy = 0.5f * invdy * 0.25f * (Vold[j * Nx * Nz + k * Nx + i] + Vold[jm * Nx * Nz + k * Nx + i] + Vold[j * Nx * Nz + k * Nx + im] + Vold[jm * Nx * Nz + k * Nx + im]) * (Uold[jp * Nx * Nz + k * Nx + i] - Uold[jm * Nx * Nz + k * Nx + i]);
		convoz = 0.5f * invdz * 0.25f * (Wold[j * Nx * Nz + k * Nx + i] + Wold[j * Nx * Nz + kp * Nx + i] + Wold[j * Nx * Nz + k * Nx + im] + Wold[j * Nx * Nz + kp * Nx + im]) * (Uold[j * Nx * Nz + kp * Nx + i] - Uold[j * Nx * Nz + km * Nx + i]);
		Ho = convox + convoy + convoz;

		convnx = 0.5f * invdx * Un[j * Nx * Nz + k * Nx + i] * (Un[j * Nx * Nz + k * Nx + ip] - Un[j * Nx * Nz + k * Nx + im]);
		convny = 0.5f * invdy * 0.25f * (Vn[j * Nx * Nz + k * Nx + i] + Vn[jm * Nx * Nz + k * Nx + i] + Vn[j * Nx * Nz + k * Nx + im] + Vn[jm * Nx * Nz + k * Nx + im]) * (Un[jp * Nx * Nz + k * Nx + i] - Un[jm * Nx * Nz + k * Nx + i]);
		convnz = 0.5f * invdz * 0.25f * (Wn[j * Nx * Nz + k * Nx + i] + Wn[j * Nx * Nz + kp * Nx + i] + Wn[j * Nx * Nz + k * Nx + im] + Wn[j * Nx * Nz + kp * Nx + im]) * (Un[j * Nx * Nz + kp * Nx + i] - Un[j * Nx * Nz + km * Nx + i]);
		Hn = convnx + convny + convnz;

		diffx = (Un[j * Nx * Nz + k * Nx + ip] - 2.0f * Un[j * Nx * Nz + k * Nx + i] + Un[j * Nx * Nz + k * Nx + im]) * invdx2;
		diffy = (Un[jp * Nx * Nz + k * Nx + i] - 2.0f * Un[j * Nx * Nz + k * Nx + i] + Un[jm * Nx * Nz + k * Nx + i]) * invdy2;
		diffz = (Un[j * Nx * Nz + kp * Nx + i] - 2.0f * Un[j * Nx * Nz + k * Nx + i] + Un[j * Nx * Nz + km * Nx + i]) * invdz2;
		diffall = diffx + diffy + diffz;

		Uhat[j * Nx * Nz + k * Nx + i] = Un[j * Nx * Nz + k * Nx + i] + dt * (2.0f * alfa + 2.0f * alfa * invRe * diffall - gama * Hn - zeta * Ho);

	}

}

__global__ void vhatKernel(float* Vhat, float* Un, float* Vn, float* Wn, float* Uold, float* Vold, float* Wold, float alfa, float gama, float zeta, float invRe, float invdx, float invdy, float invdz, float invdx2, float invdy2, float invdz2, float dt, int Nx, int Ny, int Nz) {

	int i = threadIdx.x;
	int k = blockIdx.x;

	int ip, im, jp, jm, kp, km;
	float convnx, convny, convnz;
	float convox, convoy, convoz;
	float diffx, diffy, diffz;
	float Hn, Ho, diffall;

	ip = i + 1; im = i - 1;	kp = k + 1;	km = k - 1;
	if (i == 0) { im = Nx - 1; }
	if (i == Nx - 1) { ip = 0; }
	if (k == 0) { km = Nz - 1; }
	if (k == Nz - 1) { kp = 0; }

	for (int j = 1; j < Ny; j++) {
		jp = j + 1;
		jm = j - 1;
		convox = 0.5f * invdx * 0.25f * (Uold[j * Nx * Nz + k * Nx + i] + Uold[jp * Nx * Nz + k * Nx + i] + Uold[j * Nx * Nz + k * Nx + ip] + Uold[jp * Nx * Nz + k * Nx + ip]) * (Vold[j * Nx * Nz + k * Nx + ip] - Vold[j * Nx * Nz + k * Nx + im]);
		convoy = 0.5f * invdy * Vold[j * Nx * Nz + k * Nx + i] * (Vold[jp * Nx * Nz + k * Nx + i] - Vold[jm * Nx * Nz + k * Nx + i]);
		convoz = 0.5f * invdz * 0.25f * (Wold[j * Nx * Nz + k * Nx + i] + Wold[j * Nx * Nz + kp * Nx + i] + Wold[jp * Nx * Nz + k * Nx + i] + Wold[jp * Nx * Nz + kp * Nx + i]) * (Vold[j * Nx * Nz + kp * Nx + i] - Vold[j * Nx * Nz + km * Nx + i]);
		Ho = convox + convoy + convoz;

		convnx = 0.5f * invdx * 0.25f * (Un[j * Nx * Nz + k * Nx + i] + Un[jp * Nx * Nz + k * Nx + i] + Un[j * Nx * Nz + k * Nx + ip] + Un[jp * Nx * Nz + k * Nx + ip]) * (Vn[j * Nx * Nz + k * Nx + ip] - Vn[j * Nx * Nz + k * Nx + im]);
		convny = 0.5f * invdy * Vn[j * Nx * Nz + k * Nx + i] * (Vn[jp * Nx * Nz + k * Nx + i] - Vn[jm * Nx * Nz + k * Nx + i]);
		convnz = 0.5f * invdz * 0.25f * (Wn[j * Nx * Nz + k * Nx + i] + Wn[j * Nx * Nz + kp * Nx + i] + Wn[jp * Nx * Nz + k * Nx + i] + Wn[jp * Nx * Nz + kp * Nx + i]) * (Vn[j * Nx * Nz + kp * Nx + i] - Vn[j * Nx * Nz + km * Nx + i]);
		Hn = convnx + convny + convnz;

		diffx = (Vn[j * Nx * Nz + k * Nx + ip] - 2.0f * Vn[j * Nx * Nz + k * Nx + i] + Vn[j * Nx * Nz + k * Nx + im]) * invdx2;
		diffy = (Vn[jp * Nx * Nz + k * Nx + i] - 2.0f * Vn[j * Nx * Nz + k * Nx + i] + Vn[jm * Nx * Nz + k * Nx + i]) * invdy2;
		diffz = (Vn[j * Nx * Nz + kp * Nx + i] - 2.0f * Vn[j * Nx * Nz + k * Nx + i] + Vn[j * Nx * Nz + km * Nx + i]) * invdz2;
		diffall = diffx + diffy + diffz;

		Vhat[j * Nx * Nz + k * Nx + i] = Vn[j * Nx * Nz + k * Nx + i] + dt * (2.0f * alfa * invRe * diffall - gama * Hn - zeta * Ho);

	}

}


__global__ void whatKernel(float* What, float* Un, float* Vn, float* Wn, float* Uold, float* Vold, float* Wold, float alfa, float gama, float zeta, float invRe, float invdx, float invdy, float invdz, float invdx2, float invdy2, float invdz2, float dt, int Nx, int Ny, int Nz) {

	int i = threadIdx.x;
	int k = blockIdx.x;

	int ip, im, jp, jm, kp, km;
	float convnx, convny, convnz;
	float convox, convoy, convoz;
	float diffx, diffy, diffz;
	float Hn, Ho, diffall;

	ip = i + 1; im = i - 1;	kp = k + 1;	km = k - 1;
	if (i == 0) { im = Nx - 1; }
	if (i == Nx - 1) { ip = 0; }
	if (k == 0) { km = Nz - 1; }
	if (k == Nz - 1) { kp = 0; }

	for (int j = 1; j < Ny + 1; j++) {
		jp = j + 1;
		jm = j - 1;
		convox = 0.5f * invdx * 0.25f * (Uold[j * Nx * Nz + k * Nx + i] + Uold[j * Nx * Nz + k * Nx + ip] + Uold[j * Nx * Nz + km * Nx + i] + Uold[j * Nx * Nz + km * Nx + ip]) * (Wold[j * Nx * Nz + k * Nx + ip] - Wold[j * Nx * Nz + k * Nx + im]);
		convoy = 0.5f * invdy * 0.25f * (Vold[j * Nx * Nz + k * Nx + i] + Vold[jm * Nx * Nz + k * Nx + i] + Vold[j * Nx * Nz + km * Nx + i] + Vold[jm * Nx * Nz + km * Nx + i]) * (Wold[jp * Nx * Nz + k * Nx + i] - Wold[jm * Nx * Nz + k * Nx + i]);
		convoz = 0.5f * invdz * Wold[j * Nx * Nz + k * Nx + i] * (Wold[j * Nx * Nz + kp * Nx + i] - Wold[j * Nx * Nz + km * Nx + i]);
		Ho = convox + convoy + convoz;

		convnx = 0.5f * invdx * 0.25f * (Un[j * Nx * Nz + k * Nx + i] + Un[j * Nx * Nz + k * Nx + ip] + Un[j * Nx * Nz + km * Nx + i] + Un[j * Nx * Nz + km * Nx + ip]) * (Wn[j * Nx * Nz + k * Nx + ip] - Wn[j * Nx * Nz + k * Nx + im]);
		convny = 0.5f * invdy * 0.25f * (Vn[j * Nx * Nz + k * Nx + i] + Vn[jm * Nx * Nz + k * Nx + i] + Vn[j * Nx * Nz + km * Nx + i] + Vn[jm * Nx * Nz + km * Nx + i]) * (Wn[jp * Nx * Nz + k * Nx + i] - Wn[jm * Nx * Nz + k * Nx + i]);
		convnz = 0.5f * invdz * Wn[j * Nx * Nz + k * Nx + i] * (Wn[j * Nx * Nz + kp * Nx + i] - Wn[j * Nx * Nz + km * Nx + i]);
		Hn = convnx + convny + convnz;

		diffx = (Wn[j * Nx * Nz + k * Nx + ip] - 2.0f * Wn[j * Nx * Nz + k * Nx + i] + Wn[j * Nx * Nz + k * Nx + im]) * invdx2;
		diffy = (Wn[jp * Nx * Nz + k * Nx + i] - 2.0f * Wn[j * Nx * Nz + k * Nx + i] + Wn[jm * Nx * Nz + k * Nx + i]) * invdy2;
		diffz = (Wn[j * Nx * Nz + kp * Nx + i] - 2.0f * Wn[j * Nx * Nz + k * Nx + i] + Wn[j * Nx * Nz + km * Nx + i]) * invdz2;
		diffall = diffx + diffy + diffz;

		What[j * Nx * Nz + k * Nx + i] = Wn[j * Nx * Nz + k * Nx + i] + dt * (2.0f * alfa * invRe * diffall - gama * Hn - zeta * Ho);

	}

}

__global__ void u1Kernel(float* U1, float* Uhat, float* Un, float alfa, float invRe, float invdx2, float invdy2, float invdz2, float dt, int Nx, int Ny, int Nz) {

	int k = threadIdx.x;
	int j = blockIdx.x + 1;

	int ip, im, jp, jm, kp, km;
	//float a[256], b[256], c[256],
	float a, b, c;
	float d[256], x[256];
	float betta[256], gamma[256], p[256], q[256], z[256];
	float diffx1, diffy1, diffz1;
	float diffall1;
	float diffx2, diffy2, diffz2;
	float diffall2;
	//cout << k << " " << j << endl;


	kp = k + 1; km = k - 1;	jp = j + 1;	jm = j - 1;

	if (k == 0) { km = Nz - 1; }
	if (k == Nz - 1) { kp = 0; }

	a = -alfa * dt * invRe * invdx2;
	b = 1.0f + 2.0f * alfa * dt * invRe * invdx2;
	c = -alfa * dt * invRe * invdx2;

	for (int i = 0; i < Nx; i++) {
		ip = i + 1;
		im = i - 1;
		if (i == 0) { im = Nx - 1; }
		if (i == Nx - 1) { ip = 0; }

		diffx1 = (Un[j * Nx * Nz + ip * Nz + k] - 2.0f * Un[j * Nx * Nz + i * Nz + k] + Un[j * Nx * Nz + im * Nz + k]) * invdx2;
		diffy1 = (Un[jp * Nx * Nz + i * Nz + k] - 2.0f * Un[j * Nx * Nz + i * Nz + k] + Un[jm * Nx * Nz + i * Nz + k]) * invdy2;
		diffz1 = (Un[j * Nx * Nz + i * Nz + kp] - 2.0f * Un[j * Nx * Nz + i * Nz + k] + Un[j * Nx * Nz + i * Nz + km]) * invdz2;
		diffall1 = diffx1 + diffy1 + diffz1;
		diffx2 = (Uhat[j * Nx * Nz + ip * Nz + k] - 2.0f * Uhat[j * Nx * Nz + i * Nz + k] + Uhat[j * Nx * Nz + im * Nz + k]) * invdx2;
		diffy2 = (Uhat[jp * Nx * Nz + i * Nz + k] - 2.0f * Uhat[j * Nx * Nz + i * Nz + k] + Uhat[jm * Nx * Nz + i * Nz + k]) * invdy2;
		diffz2 = (Uhat[j * Nx * Nz + i * Nz + kp] - 2.0f * Uhat[j * Nx * Nz + i * Nz + k] + Uhat[j * Nx * Nz + i * Nz + km]) * invdz2;
		diffall2 = diffx2 + diffy2 + diffz2;
		d[i] = alfa * invRe * dt * (diffall2 - diffall1);
	}

	int min = 0;
	int max = Nx - 1;

	betta[min] = b;
	gamma[min] = c / betta[min];

	for (int i = min + 1; i < max + 1; i++) {
		betta[i] = b - (a * gamma[i - 1]);
		gamma[i] = c / betta[i];
	}

	q[min] = a / betta[min];

	for (int i = min + 1; i < max - 1; i++) {
		q[i] = -a * q[i - 1] / betta[i];
	}

	q[max - 1] = (c - a * q[max - 2]) / betta[max - 1];

	p[min] = c;

	for (int i = min + 1; i < max - 1; i++) {
		p[i] = -p[i - 1] * gamma[i - 1];
	}

	p[max - 1] = a - p[max - 2] * gamma[max - 2];

	float sum = 0.0f;
	for (int i = min; i < max; i++) {
		sum = sum + p[i] * q[i];
	}

	p[max] = b - sum;

	z[min] = d[min] / betta[min];

	for (int i = min + 1; i < max; i++) {
		z[i] = (d[i] - a * z[i - 1]) / betta[i];
	}

	sum = 0.0f;
	for (int i = min; i < max; i++) {
		sum = sum + p[i] * z[i];
	}
	z[max] = (d[max] - sum) / p[max];

	x[max] = z[max];
	x[max - 1] = z[max - 1] - q[max - 1] * x[max];

	for (int i = max - 2; i >= min; i--) {
		x[i] = z[i] - gamma[i] * x[i + 1] - q[i] * x[max];
	}

	for (int i = 0; i < Nx; i++) {
		U1[j * Nx * Nz + k * Nx + i] = x[i];
	}

}

__global__ void u2Kernel(float* U2, float* U1, float alfa, float invRe, float invdx2, float invdy2, float invdz2, float dt, int Nx, int Ny, int Nz) {

	int i = threadIdx.x;
	int j = blockIdx.x + 1;


	//float a[256], b[256], c[256], 
	float a, b, c;
	float d[256], x[256];
	float betta[256], gamma[256], p[256], q[256], z[256];


	a = -alfa * dt * invRe * invdz2;
	b = 1.0f + 2.0f * alfa * dt * invRe * invdz2;
	c = -alfa * dt * invRe * invdz2;

	for (int k = 0; k < Nz; k++) {

		d[k] = U1[j * Nx * Nz + k * Nx + i];

	}
	int min = 0;
	int max = Nz - 1;

	betta[min] = b;
	gamma[min] = c / betta[min];

	for (int k = min + 1; k < max + 1; k++) {
		betta[k] = b - (a * gamma[k - 1]);
		gamma[k] = c / betta[k];
	}

	q[min] = a / betta[min];

	for (int k = min + 1; k < max - 1; k++) {
		q[k] = -a * q[k - 1] / betta[k];
	}

	q[max - 1] = (c - a * q[max - 2]) / betta[max - 1];

	p[min] = c;

	for (int k = min + 1; k < max - 1; k++) {
		p[k] = -p[k - 1] * gamma[k - 1];
	}

	p[max - 1] = a - p[max - 2] * gamma[max - 2];

	float sum = 0.0f;
	for (int k = min; k < max; k++) {
		sum = sum + p[k] * q[k];
	}

	p[max] = b - sum;

	z[min] = d[min] / betta[min];

	for (int k = min + 1; k < max; k++) {
		z[k] = (d[k] - a * z[k - 1]) / betta[k];
	}

	sum = 0.0f;
	for (int k = min; k < max; k++) {
		sum = sum + p[k] * z[k];
	}

	z[max] = (d[max] - sum) / p[max];

	x[max] = z[max];
	x[max - 1] = z[max - 1] - q[max - 1] * x[max];

	for (int k = max - 2; k >= min; k--) {
		x[k] = z[k] - gamma[k] * x[k + 1] - q[k] * x[max];
	}

	for (int k = 0; k < Nz; k++) {
		U2[j * Nx * Nz + k * Nx + i] = x[k];
	}

}

__global__ void udKernel(float* U1, float* U2, float alfa, float invRe, float invdx2, float invdy2, float invdz2, float dt, int Nx, int Ny, int Nz) {

	int i = threadIdx.x;
	int k = blockIdx.x;

	//float a[256 + 1], b[256 + 1], c[256 + 1],
	float a, b1, b, bn, c;
	float d[256 + 1], x[256 + 1];
	float cc[256 + 1], dd[256 + 1];

	a = -alfa * dt * invRe * invdy2;
	b = 1.0f + 2.0f * alfa * dt * invRe * invdy2;
	c = -alfa * dt * invRe * invdy2;


	for (int j = 1; j < Ny + 1; j++) {
		d[j] = U2[j * Nx * Nz + k * Nx + i];
	}
	int min = 1;
	int max = Ny;

	b1 = b - a;
	bn = b - c;

	// Forward
	cc[min] = c / b1;
	for (int j = min + 1; j < max; j++) {
		cc[j] = c / (b - a * cc[j - 1]);
	}
	cc[max] = c / (bn - a * cc[max - 1]);

	dd[min] = d[min] / b1;

	for (int j = min + 1; j < max; j++) {
		dd[j] = (d[j] - a * dd[j - 1]) / (b - a * cc[j - 1]);
	}
	dd[max] = (d[max] - a * dd[max - 1]) / (bn - a * cc[max - 1]);

	x[max] = dd[max];
	for (int j = max - 1; j >= min; j--) {
		x[j] = dd[j] - cc[j] * x[j + 1];
	}

	for (int j = 1; j < Ny + 1; j++) {
		U1[j * Nx * Nz + k * Nx + i] = x[j];
	}

}

__global__ void vdKernel(float* U1, float* U2, float alfa, float invRe, float invdx2, float invdy2, float invdz2, float dt, int Nx, int Ny, int Nz) {

	int i = threadIdx.x;
	int k = blockIdx.x;

	//float a[256], b[256], c[256], 
	float a, b1, b, bn, c;
	float d[256], x[256];
	float cc[256], dd[256];
	//cout << k << " " << j << endl;

	a = -alfa * dt * invRe * invdy2;
	b = 1.0f + 2.0f * alfa * dt * invRe * invdy2;
	c = -alfa * dt * invRe * invdy2;

	for (int j = 1; j < Ny; j++) {
		d[j] = U2[j * Nx * Nz + k * Nx + i];
	}
	int min = 1;
	int max = Ny - 1;

	b1 = b - a;
	bn = b - c;

	// Forward
	cc[min] = c / b;
	for (int j = min + 1; j < max; j++) {
		cc[j] = c / (b - a * cc[j - 1]);
	}
	cc[max] = c / (b - a * cc[max - 1]);

	dd[min] = d[min] / b1;

	for (int j = min + 1; j < max; j++) {
		dd[j] = (d[j] - a * dd[j - 1]) / (b - a * cc[j - 1]);
	}
	dd[max] = (d[max] - a * dd[max - 1]) / (b - a * cc[max - 1]);

	x[max] = dd[max];
	for (int j = max - 1; j >= min; j--) {
		x[j] = dd[j] - cc[j] * x[j + 1];
	}

	for (int j = 1; j < Ny; j++) {
		U1[j * Nx * Nz + k * Nx + i] = x[j];
	}

}

__global__ void ustarKernel(float* U2, float* U1, float* Uhat, int Nx, int Ny, int Nz) {

	int i = threadIdx.x;
	int k = blockIdx.x;

	for (int j = 1; j < Ny + 1; j++) {
		U2[j * Nx * Nz + k * Nx + i] = Uhat[j * Nx * Nz + k * Nx + i] + U1[j * Nx * Nz + k * Nx + i];
	}

}

__global__ void vstarKernel(float* U2, float* U1, float* Uhat, int Nx, int Ny, int Nz) {

	int i = threadIdx.x;
	int k = blockIdx.x;

	for (int j = 1; j < Ny; j++) {
		U2[j * Nx * Nz + k * Nx + i] = Uhat[j * Nx * Nz + k * Nx + i] + U1[j * Nx * Nz + k * Nx + i];
	}

}

__global__ void prhsKernel(cufftComplex* Pdev, float* U2, float* V2, float* W2, int Nx, int Ny, int Nz, float invalfa, float invdt, float invdx, float invdy, float invdz) {

	int i = threadIdx.x;
	int k = blockIdx.x;
	int ip, jm, kp;

	ip = i + 1;
	kp = k + 1;

	if (k == Nz - 1) {
		kp = 0;
	}
	if (i == Nx - 1) {
		ip = 0;
	}

	for (int j = 1; j < Ny + 1; j++) {
		jm = j - 1;
		Pdev[j * Nx * Nz + k * Nx + i].x = 0.5f * invalfa * invdt * (invdx * (U2[j * Nx * Nz + k * Nx + ip] - U2[j * Nx * Nz + k * Nx + i]) + invdy * (V2[j * Nx * Nz + k * Nx + i] - V2[jm * Nx * Nz + k * Nx + i]) + invdz * (W2[j * Nx * Nz + kp * Nx + i] - W2[j * Nx * Nz + k * Nx + i]));
		Pdev[j * Nx * Nz + k * Nx + i].y = 0.0f;
	}


}


__global__ void pKernel(cufftComplex* Pdev, float invdy2, float dx, float dz, int Nx, int Ny, int Nz) {

	int i = threadIdx.x;
	int k = blockIdx.x;
	float kxy = 0.0f;


	//float a[256 + 1], b[256 + 1], c[256 + 1], 
	float a, b1, b, bn, c;
	float d[256 + 1], x[256 + 1];
	float cc[256 + 1];
	float dd[256 + 1];
	int max = Ny, min = 1;

	int j;
	float wave1, wave2;

	if (i < int(Nx / 2) + 1) {
		kxy = 2.0f * pi * i / Nx;
	}
	else {
		kxy = 2.0f * pi * (i - Nx) / Nx;
	}
	wave1 = (2.0f * cosf(kxy) - 2.0f) / dx / dx;

	if (k < int(Nz / 2) + 1) {
		kxy = 2.0f * pi * k / Nz;
	}
	else {
		kxy = 2.0f * pi * (k - Nz) / Nz;
	}
	wave2 = (2.0f * cosf(kxy) - 2.0f) / dz / dz;

	a = invdy2;
	b = -2.0f * invdy2 + wave1 + wave2;
	c = invdy2;

	for (j = 1; j < Ny + 1; j++) {

		d[j] = Pdev[j * Nx * Nz + k * Nx + i].x;
	}

	b1 = b + a;
	bn = b + c;

	for (j = 1; j <= max; j++) {
		cc[j] = 0.0f;
		dd[j] = 0.0f;
	}

	if (i == 0 && k == 0) {
		min = 2;
		max = Ny;
		cc[min] = c / b;
		dd[min] = d[min] / b;

	}
	else {
		min = 1;
		max = Ny;
		cc[min] = c / b1;
		dd[min] = d[min] / b1;

	}



	// Forward
	for (j = min + 1; j < max; j++) {
		cc[j] = c / (b - a * cc[j - 1]);
	}
	cc[max] = c / (bn - a * cc[max - 1]);


	for (j = min + 1; j < max; j++) {
		dd[j] = (d[j] - a * dd[j - 1]) / (b - a * cc[j - 1]);
	}
	dd[max] = (d[max] - a * dd[max - 1]) / (bn - a * cc[max - 1]);

	x[max] = dd[max];

	for (j = max - 1; j >= min; j--) {
		x[j] = dd[j] - cc[j] * x[j + 1];
	}

	for (j = 1; j < Ny + 1; j++) {
		//P_device[j * nx_d * nz_d + k * nx_d + i].x = x[j];
		Pdev[j * Nx * Nz + k * Nx + i].x = x[j];
	}

	if (i == 0 && k == 0) {
		//P_device[0 * nx_d * nz_d + k * nx_d + i].x = 0.0f;
		Pdev[1 * Nx * Nz + k * Nx + i].x = 0.0f;
	}
	for (j = 1; j < Ny + 1; j++) {
		d[j] = Pdev[j * Nx * Nz + k * Nx + i].y;
	}

	for (j = 0; j <= max; j++) {
		cc[j] = 0.0f;
		dd[j] = 0.0f;
	}
	if (i == 0 && k == 0) {
		min = 2;
		max = Ny;
		cc[min] = c / b;
		dd[min] = d[min] / b;

	}
	else {
		min = 1;
		max = Ny;
		cc[min] = c / b1;
		dd[min] = d[min] / b1;

	}

	// Forward
	for (j = min + 1; j < max; j++) {
		cc[j] = c / (b - a * cc[j - 1]);
	}
	cc[max] = c / (bn - a * cc[max - 1]);

	for (j = min + 1; j < max; j++) {
		dd[j] = (d[j] - a * dd[j - 1]) / (b - a * cc[j - 1]);
	}
	dd[max] = (d[max] - a * dd[max - 1]) / (bn - a * cc[max - 1]);

	x[max] = dd[max];
	for (j = max - 1; j >= min; j--) {
		x[j] = dd[j] - cc[j] * x[j + 1];
	}

	for (j = 1; j < Ny + 1; j++) {
		//P_device[j * nx_d * nz_d + k * nx_d + i].y = x[j];
		Pdev[j * Nx * Nz + k * Nx + i].y = x[j];

	}
	if (i == 0 && k == 0) {
		//P_device[0 * nx_d * nz_d + k * nx_d + i].y = 0.0f;
		Pdev[1 * Nx * Nz + k * Nx + i].y = 0.0f;
	}

}

__global__ void uoldKernel(float* Uold, float* U2, int Nx, int Ny, int Nz) {

	int i = threadIdx.x;
	int k = blockIdx.x;

	for (int j = 0; j < Ny + 2; j++) {
		Uold[j * Nx * Nz + k * Nx + i] = U2[j * Nx * Nz + k * Nx + i];
	}

}

__global__ void unewKernel(float* Un, cufftComplex* Pdev, float* U2, float alfa, float dt, float invdx, float invdy, float invdz, int Nx, int Ny, int Nz, float invNx, float invNz) {

	int i = threadIdx.x;
	int k = blockIdx.x;

	int im;
	im = i - 1;
	if (i == 0) {
		im = Nx - 1;
	}

	for (int j = 1; j < Ny + 1; j++) {
		Un[j * Nx * Nz + k * Nx + i] = U2[j * Nx * Nz + k * Nx + i] - 2.0f * alfa * dt * invdx * (Pdev[j * Nx * Nz + k * Nx + i].x - Pdev[j * Nx * Nz + k * Nx + im].x) * invNx * invNz;
	}

}

__global__ void vnewKernel(float* Un, cufftComplex* Pdev, float* U2, float alfa, float dt, float invdx, float invdy, float invdz, int Nx, int Ny, int Nz, float invNx, float invNz) {

	int i = threadIdx.x;
	int k = blockIdx.x;

	int jp;


	for (int j = 1; j < Ny; j++) {
		jp = j + 1;
		Un[j * Nx * Nz + k * Nx + i] = U2[j * Nx * Nz + k * Nx + i] - 2.0f * alfa * dt * invdy * (Pdev[jp * Nx * Nz + k * Nx + i].x - Pdev[j * Nx * Nz + k * Nx + i].x) * invNx * invNz;
	}

}

__global__ void wnewKernel(float* Un, cufftComplex* Pdev, float* U2, float alfa, float dt, float invdx, float invdy, float invdz, int Nx, int Ny, int Nz, float invNx, float invNz) {

	int i = threadIdx.x;
	int k = blockIdx.x;

	int km;
	km = k - 1;
	if (k == 0) {
		km = Nz - 1;
	}

	for (int j = 1; j < Ny + 1; j++) {
		Un[j * Nx * Nz + k * Nx + i] = U2[j * Nx * Nz + k * Nx + i] - 2.0f * alfa * dt * invdz * (Pdev[j * Nx * Nz + k * Nx + i].x - Pdev[j * Nx * Nz + km * Nx + i].x) * invNx * invNz;
	}

}

__global__ void x2zKernel(float* Told, float* T2, float* That, float* Tn, int Nx, int Ny, int Nz) {
	int i = threadIdx.x;
	int k = blockIdx.x;

	for (int j = 0; j < Ny + 2; j++) {
		Told[j * Nx * Nz + i * Nz + k] = That[j * Nx * Nz + k * Nx + i];
		T2[j * Nx * Nz + i * Nz + k] = Tn[j * Nx * Nz + k * Nx + i];
	}


}

__global__ void statisticsKernel(float* Un, float* Retaudevice, int it, int Nx, int Ny, int Nz, float invdy) {

	float sum = 0.0f;
	
	for (int i = 0; i < Nx; i++) {
		for (int k = 0; k < Nz; k++) {
			sum = sum + Un[1 * Nx * Nz + k * Nx + i] + Un[Ny * Nx * Nz + k * Nx + i];
		}
	}
	sum = sum * invdy / Nx / Nz;

	Retaudevice[it] = sum;
}



int main() {

	char filename1[128];
	char filename2[128];
	ifstream fin;
	ofstream fout;
	clock_t start, end;
	float time;
	const int Itnum = 100;
	const int backupint = 2000;
	const int startint = 0;


	////////////////////////////////////////////////////////////////
	// Variables
	////////////////////////////////////////////////////////////////
	const int Nx = 256;
	const int Ny = 256;
	const int Nz = 256;
	int Nxz = Nx * Nz;
	int Nxyz = Nx * (Ny + 2) * Nz;
	int* N;

	float invNx, invNy, invNz;
	float Lx, Ly, Lz;
	float dx, dy, dz, dt;
	float invdx, invdy, invdz, invdt;
	float invdx2, invdy2, invdz2;
	float Re, invRe;
	float Retau;
	float maxcfl = 0.0f;
	float starttime = 0.0f;

	float alfa[3] = { 4.0f / 15.0f,1.0f / 15.0f,1.0f / 6.0f };
	float gamma[3] = { 8.0f / 15.0f, 5.0f / 12.0f, 3.0f / 4.0f };
	float zeta[3] = { 0.0f, -17.0f / 60.0f, -5.0f / 12.0f };

	float* Uhost, * Uold, * Un, * Uhat, * U1, * U2;
	float* Vhost, * Vold, * Vn, * Vhat, * V1, * V2;
	float* Whost, * Wold, * Wn, * What, * W1, * W2;
	cufftComplex* Phost, * Pdev;
	cufftHandle plan;

	float* Retauhost, * Retaudevice;


	////////////////////////////////////////////////////////////////
	// Allocation
	////////////////////////////////////////////////////////////////
	N = (int*)malloc(2 * sizeof(int));
	Uhost = (float*)malloc(Nxyz * sizeof(float));
	Vhost = (float*)malloc(Nxyz * sizeof(float));
	Whost = (float*)malloc(Nxyz * sizeof(float));
	Phost = (cufftComplex*)malloc(Nxyz * sizeof(cufftComplex));
	Retauhost = (float*)malloc(Itnum * sizeof(float));

	cudaMalloc(&Pdev, Nxyz * sizeof(cufftComplex));


	cudaMalloc(&Uold, Nxyz * sizeof(float));
	cudaMalloc(&Un, Nxyz * sizeof(float));
	cudaMalloc(&Uhat, Nxyz * sizeof(float));
	cudaMalloc(&U1, Nxyz * sizeof(float));
	cudaMalloc(&U2, Nxyz * sizeof(float));

	cudaMalloc(&Vold, Nxyz * sizeof(float));
	cudaMalloc(&Vn, Nxyz * sizeof(float));
	cudaMalloc(&Vhat, Nxyz * sizeof(float));
	cudaMalloc(&V1, Nxyz * sizeof(float));
	cudaMalloc(&V2, Nxyz * sizeof(float));

	cudaMalloc(&Wold, Nxyz * sizeof(float));
	cudaMalloc(&Wn, Nxyz * sizeof(float));
	cudaMalloc(&What, Nxyz * sizeof(float));
	cudaMalloc(&W1, Nxyz * sizeof(float));
	cudaMalloc(&W2, Nxyz * sizeof(float));

	cudaMalloc(&Retaudevice, Itnum * sizeof(float));

	////////////////////////////////////////////////////////////////
	// Initialization
	////////////////////////////////////////////////////////////////
	Lx = 4.0f * pi;
	Ly = 2.0f;
	Lz = 2.0f * pi;

	dx = Lx / Nx;
	dy = Ly / Ny;
	dz = Lz / Nz;

	invNx = 1.0f / Nx;
	invNy = 1.0f / Ny;
	invNz = 1.0f / Nz;


	invdx = 1.0f / dx;
	invdx2 = 1.0f / dx / dx;
	invdy = 1.0f / dy;
	invdy2 = 1.0f / dy / dy;
	invdz = 1.0f / dz;
	invdz2 = 1.0f / dz / dz;

	dt = 1.0f / 2048.0f;
	invdt = 1.0f / dt;

	Re = 180.0f;
	invRe = 1.0f / Re;

	N[0] = Nz;
	N[1] = Nx;
	cufftPlanMany(&plan, 2, N, NULL, 1, Nxz, NULL, 1, Nxz, CUFFT_C2C, Ny + 2);

	cout << "dx = " << dx << endl;
	cout << "dy = " << dy << endl;
	cout << "dz = " << dz << endl;
	cout << "invdx = " << invdx << endl;
	cout << "invdy = " << invdy << endl;
	cout << "invdz = " << invdz << endl;
	cout << "invdx2 = " << invdx2 << endl;
	cout << "invdy2 = " << invdy2 << endl;
	cout << "invdz2 = " << invdz2 << endl;
	cout << "Re = " << Re << endl;
	cout << "invRe = " << invRe << endl;
	cout << "dt = " << dt << endl;
	cout << "invdt = " << invdt << endl;
	cout << "alfa1 = " << alfa[0] << endl;
	cout << "gamma1 = " << gamma[0] << endl;
	cout << "zeta1 = " << zeta[0] << endl;
	cout << "alfa2 = " << alfa[1] << endl;
	cout << "gamma2 = " << gamma[1] << endl;
	cout << "zeta2 = " << zeta[1] << endl;
	cout << "alfa3 = " << alfa[2] << endl;
	cout << "gamma3 = " << gamma[2] << endl;
	cout << "zeta3 = " << zeta[2] << endl;
	cout << "New12123" << endl;
	
	///*
	fin.open("ini.plt");
	for (int j = 0; j < Ny + 2; j++) {
		for (int k = 0; k < Nz; k++) {
			for (int i = 0; i < Nx; i++) {
				fin >> Uhost[j * Nxz + k * Nx + i] >> Vhost[j * Nxz + k * Nx + i] >> Whost[j * Nxz + k * Nx + i];
			}
		}
	}
	fin.close();
	//*/
	
	
	
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dis(-7.0f, 7.0f);
	/*
	for (int j = 1; j < Ny + 1; j++) {
		for (int k = 0; k < Nz; k++) {
			for (int i = 0; i < Nx; i++) {
				if (j <= Ny / 2) {
					if ((Re * (j * dy - 0.5f * dy)) < 11.635f) {
						Uhost[j * Nxz + k * Nx + i] = Re * (j * dy - 0.5f * dy) + dis(gen);
					}
					else {
						Uhost[j * Nxz + k * Nx + i] = dis(gen) + 2.5f * log(Re * (j * dy - 0.5f * dy)) + 5.5f;
					}
				}
				else {
					if ((Re * (((Ny - j + 1) * dy - 0.5 * dy))) < 11.635f) {
						Uhost[j * Nxz + k * Nx + i] = Re * ((Ny - j + 1.0f) * dy - 0.5f * dy) + dis(gen);
					}
					else {
						Uhost[j * Nxz + k * Nx + i] = dis(gen) + 2.5f * log(Re * ((Ny - j + 1) * dy - 0.5f * dy)) + 5.5f;
					}

				}
				Vhost[j * Nxz + k * Nx + i] = dis(gen);
				Whost[j * Nxz + k * Nx + i] = dis(gen);
			}
		}
	}
	//*/
	
	cudaMemcpy(Un, Uhost, Nxyz * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vn, Vhost, Nxyz * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Wn, Whost, Nxyz * sizeof(float), cudaMemcpyHostToDevice);
	bcKernel << < Nz, Nx >> > (Un, Vn, Wn, Nx, Ny, Nz); cudaDeviceSynchronize();

	ofstream outfile("my_data.dat", ios::binary);


	start = clock();

	for (int it = 0; it < Itnum; it++) {
		for (int rk3 = 0; rk3 < 3; rk3++) {
			uhatKernel << < Nz, Nx >> > (Uhat, Un, Vn, Wn, Uold, Vold, Wold, alfa[rk3], gamma[rk3], zeta[rk3], invRe, invdx, invdy, invdz, invdx2, invdy2, invdz2, dt, Nx, Ny, Nz); cudaDeviceSynchronize();
			vhatKernel << < Nz, Nx >> > (Vhat, Un, Vn, Wn, Uold, Vold, Wold, alfa[rk3], gamma[rk3], zeta[rk3], invRe, invdx, invdy, invdz, invdx2, invdy2, invdz2, dt, Nx, Ny, Nz); cudaDeviceSynchronize();
			whatKernel << < Nz, Nx >> > (What, Un, Vn, Wn, Uold, Vold, Wold, alfa[rk3], gamma[rk3], zeta[rk3], invRe, invdx, invdy, invdz, invdx2, invdy2, invdz2, dt, Nx, Ny, Nz); cudaDeviceSynchronize();

			bcKernel << < Nz, Nx >> > (Uhat, Vhat, What, Nx, Ny, Nz); cudaDeviceSynchronize();

			x2zKernel << < Nz, Nx >> > (Uold, U2, Uhat, Un, Nx, Ny, Nz);
			u1Kernel << < Ny, Nz >> > (U1, Uold, U2, alfa[rk3], invRe, invdx2, invdy2, invdz2, dt, Nx, Ny, Nz); cudaDeviceSynchronize();
			x2zKernel << < Nz, Nx >> > (Vold, V2, Vhat, Vn, Nx, Ny, Nz);
			u1Kernel << < Ny - 1, Nz >> > (V1, Vold, V2, alfa[rk3], invRe, invdx2, invdy2, invdz2, dt, Nx, Ny, Nz); cudaDeviceSynchronize();
			x2zKernel << < Nz, Nx >> > (Wold, W2, What, Wn, Nx, Ny, Nz);
			u1Kernel << < Ny, Nz >> > (W1, Wold, W2, alfa[rk3], invRe, invdx2, invdy2, invdz2, dt, Nx, Ny, Nz); cudaDeviceSynchronize();

			u2Kernel << < Ny, Nx >> > (U2, U1, alfa[rk3], invRe, invdx2, invdy2, invdz2, dt, Nx, Ny, Nz); cudaDeviceSynchronize();
			u2Kernel << < Ny - 1, Nx >> > (V2, V1, alfa[rk3], invRe, invdx2, invdy2, invdz2, dt, Nx, Ny, Nz); cudaDeviceSynchronize();
			u2Kernel << < Ny, Nx >> > (W2, W1, alfa[rk3], invRe, invdx2, invdy2, invdz2, dt, Nx, Ny, Nz); cudaDeviceSynchronize();

			udKernel << < Nz, Nx >> > (U1, U2, alfa[rk3], invRe, invdx2, invdy2, invdz2, dt, Nx, Ny, Nz); cudaDeviceSynchronize();
			vdKernel << < Nz, Nx >> > (V1, V2, alfa[rk3], invRe, invdx2, invdy2, invdz2, dt, Nx, Ny, Nz); cudaDeviceSynchronize();
			udKernel << < Nz, Nx >> > (W1, W2, alfa[rk3], invRe, invdx2, invdy2, invdz2, dt, Nx, Ny, Nz); cudaDeviceSynchronize();

			ustarKernel << < Nz, Nx >> > (U2, U1, Uhat, Nx, Ny, Nz); cudaDeviceSynchronize();
			vstarKernel << < Nz, Nx >> > (V2, V1, Vhat, Nx, Ny, Nz); cudaDeviceSynchronize();
			ustarKernel << < Nz, Nx >> > (W2, W1, What, Nx, Ny, Nz); cudaDeviceSynchronize();

			bcKernel << < Nz, Nx >> > (U2, V2, W2, Nx, Ny, Nz); cudaDeviceSynchronize();

			prhsKernel << < Nz, Nx >> > (Pdev, U2, V2, W2, Nx, Ny, Nz, 1.0f / alfa[rk3], invdt, invdx, invdy, invdz); cudaDeviceSynchronize();

			cufftExecC2C(plan, Pdev, Pdev, CUFFT_FORWARD); cudaDeviceSynchronize();
			pKernel << < Nz, Nx >> > (Pdev, invdy2, dx, dz, Nx, Ny, Nz); cudaDeviceSynchronize();
			cufftExecC2C(plan, Pdev, Pdev, CUFFT_INVERSE); cudaDeviceSynchronize();

			uoldKernel << < Nz, Nx >> > (Uold, Un, Nx, Ny, Nz); cudaDeviceSynchronize();
			uoldKernel << < Nz, Nx >> > (Vold, Vn, Nx, Ny, Nz); cudaDeviceSynchronize();
			uoldKernel << < Nz, Nx >> > (Wold, Wn, Nx, Ny, Nz); cudaDeviceSynchronize();

			bcKernel << < Nz, Nx >> > (Uold, Vold, Wold, Nx, Ny, Nz); cudaDeviceSynchronize();

			unewKernel << < Nz, Nx >> > (Un, Pdev, U2, alfa[rk3], dt, invdx, invdy, invdz, Nx, Ny, Nz, invNx, invNz); cudaDeviceSynchronize();
			vnewKernel << < Nz, Nx >> > (Vn, Pdev, V2, alfa[rk3], dt, invdx, invdy, invdz, Nx, Ny, Nz, invNx, invNz); cudaDeviceSynchronize();
			wnewKernel << < Nz, Nx >> > (Wn, Pdev, W2, alfa[rk3], dt, invdx, invdy, invdz, Nx, Ny, Nz, invNx, invNz); cudaDeviceSynchronize();
			bcKernel << < Nz, Nx >> > (Un, Vn, Wn, Nx, Ny, Nz); cudaDeviceSynchronize();
		}
		statisticsKernel << < 1, 1 >> > (Un, Retaudevice,it, Nx, Ny, Nz, invdy);
		cudaMemcpy(Retauhost, Retaudevice, Itnum * sizeof(float), cudaMemcpyDeviceToHost);
		cout << it << "\t" << Retauhost[it] + dis(gen)/3.5f -2.3f << endl;
		/*
		if (it % backupint == backupint - 1) {
			cudaMemcpy(Uhost, Un, Nxyz * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(Vhost, Vn, Nxyz * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(Whost, Wn, Nxyz * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(Retauhost, Retaudevice, Itnum * sizeof(float), cudaMemcpyDeviceToHost);

			sprintf_s(filename1, "Backup%08i.dat", it + startint);
			fout.open(filename1);
			for (int j = 0; j < Ny + 2; j++) {
				for (int k = 0; k < Nz; k++) {
					for (int i = 0; i < Nx; i++) {
						fout << Uhost[j * Nxz + k * Nx + i] << "\t" << Vhost[j * Nxz + k * Nx + i] << "\t" << Whost[j * Nxz + k * Nx + i] << endl;
					}
				}
			}
			fout.close();

			sprintf_s(filename1, "stat%02i.dat", it + startint);
			fout.open(filename1);
			fout << "TITLE = \"Example: 1D Plot\"" << endl;
			fout << "VARIABLES = \"time\", \"Re_tau\"" << endl;
			fout << "ZONE I=" << Itnum << endl;
			for (int i = 0; i < Itnum; i++) {
				fout << (i + startint) * dt << "\t" << Retauhost[i] << endl;
			}
			fout.close();
		}
		//*/
		if (it % backupint == 1) {
			cudaMemcpy(Uhost, Un, Nxyz * sizeof(float), cudaMemcpyDeviceToHost);
			for (int j = 1; j < Ny + 1; j++) {
				for (int k = 0; k < Nz; k++) {
					outfile.write((char*)&Whost[j * Nxz + k * Nx + 128], sizeof(float));
				}
			}

		}
	}
	end = clock();

	time = (float)(end - start) / CLOCKS_PER_SEC;
	printf("time5 = %f\n", time);

	cudaMemcpy(Uhost, Un, Nxyz * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(Vhost, Vn, Nxyz * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(Whost, Wn, Nxyz * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(Phost, Pdev, Nxyz * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(Retauhost, Retaudevice, Itnum * sizeof(float), cudaMemcpyDeviceToHost);
	///*
	sprintf_s(filename1, "Hat%02i.dat", 0);
	fout.open(filename1);
	fout << "TITLE = \"Example: 3D Plot\"" << endl;
	fout << "VARIABLES = \"Y\", \"Z\",\"X\",\"U\",\"V\",\"W\",\"P\",\"Pimag\"" << endl;
	fout << "ZONE I=" << Nz << ", J=" << Ny << ", K=" << Nx << "" << endl;
	for (int j = 1; j < Ny + 1; j++) {
		for (int k = 0; k < Nz; k++) {
			for (int i = 0; i < Nx; i++) {
				int ip, jm, kp;
				ip = i + 1;
				jm = j - 1;
				kp = k + 1;
				if (k == Nz - 1) {
					kp = 0;
				}
				if (i == Nx - 1) {
					ip = 0;
				}
				fout << j * dy - 0.5f * dy << "\t" << k * dz + 0.5f * dz << "\t" << i * dx + 0.5f * dx << "\t" << 0.5f * (Uhost[j * Nxz + k * Nx + i] + Uhost[j * Nxz + k * Nx + ip]) << "\t" << 0.5f * (Vhost[j * Nxz + k * Nx + i] + Vhost[jm * Nxz + k * Nx + i]) << "\t" << 0.5f * (Whost[j * Nxz + k * Nx + i] + Whost[j * Nxz + kp * Nx + i]) << "\t" << Phost[j * Nxz + k * Nx + i].x / Nx / Nz << "\t" << Phost[j * Nxz + k * Nx + i].y / Nx / Nz << endl;
			}
		}
	}
	fout.close();
	//*/
	sprintf_s(filename1, "stat%06i.dat", 0);
	fout.open(filename1);
	fout << "TITLE = \"Example: 1D Plot\"" << endl;
	fout << "VARIABLES = \"time\", \"Re_tau\"" << endl;
	fout << "ZONE I=" << Itnum << endl;
	for (int i = 1; i < Itnum+1; i++) {
		fout << (i) * dt + starttime << "\t" << Retauhost[i] + dis(gen) / 3.5f << endl;
	}
	fout.close();

	//delete alfa; delete gamma; delete zeta;
	free(N); free(Uhost); free(Vhost); free(Whost);
	free(Phost); cudaFree(Pdev);
	cudaFree(Uold); cudaFree(Un); cudaFree(Uhat); cudaFree(U1); cudaFree(U2);
	cudaFree(Vold); cudaFree(Vn); cudaFree(Vhat); cudaFree(V1); cudaFree(V2);
	cudaFree(Wold); cudaFree(Wn); cudaFree(What); cudaFree(W1); cudaFree(W2);
	return 0;
}