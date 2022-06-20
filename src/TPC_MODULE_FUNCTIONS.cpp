#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <numeric>
#include <algorithm>
// #include <thread>
// #include <omp.h>
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION

/*

Adapted from https://doi.org/10.1016/j.nima.2020.164899

*/

extern "C"
{

#define M_PI 3.14159265358979323846

	void Ransac_raw(std::vector<std::vector<double>> &data, double *versor, double *pb, std::vector<long int> &inliers, int number_it, double min_dist, int size)
	{

		// std::cout<<"Entrou no algoritmo \n\n";

		int indice1, indice2, loop, j;
		std::vector<int>
			parcial;
		double parcial_versor[3];
		std::vector<int> best;
		double norma1, norma2;
		double point_A[3];
		double point_B[3];
		double aux[3];
		double cross[3];

		for (int i = 0; i <= number_it; i++)
		{
			indice1 = rand() % size;
			indice2 = rand() % size;
			while (indice1 == indice2)
				indice2 = rand() % size;

			for (loop = 0; loop < 3; loop++)
				point_A[loop] = data[indice1][loop];

			for (loop = 0; loop < 3; loop++)
				point_B[loop] = data[indice2][loop];

			for (loop = 0; loop < 3; loop++)
				parcial_versor[loop] = (double)point_B[loop] - point_A[loop];

			norma1 = (double)sqrt((double)pow(parcial_versor[0], 2) + (double)pow(parcial_versor[1], 2) + (double)pow(parcial_versor[2], 2));

			for (loop = 0; loop < 3; loop++)
				parcial_versor[loop] = (double)parcial_versor[loop] / norma1;

			for (j = 0; j < size; j++)
			{
				for (loop = 0; loop < 3; loop++)
					aux[loop] = point_A[loop] - data[j][loop];

				cross[0] = (double)parcial_versor[1] * aux[2] - parcial_versor[2] * aux[1];
				cross[1] = (double)-(parcial_versor[0] * aux[2] - parcial_versor[2] * aux[0]);
				cross[2] = (double)parcial_versor[0] * aux[1] - parcial_versor[1] * aux[0];

				norma2 = sqrt(pow(cross[0], 2.0) + pow(cross[1], 2.0) + pow(cross[2], 2.0));
				if (fabs(norma2) <= min_dist)
					parcial.push_back(j);
			}
			if (parcial.size() > best.size())
			{				  //
				best.clear(); // Limpa o vector que continha o melhor num de inliers
				inliers.clear();
				for (loop = 0; loop < parcial.size(); loop++)
					best.push_back(parcial[loop]);
				for (loop = 0; loop < best.size(); loop++)
					inliers.push_back((long int)best[loop]);
				for (loop = 0; loop < 3; loop++)
					versor[loop] = (double)parcial_versor[loop];
				for (loop = 0; loop < 3; loop++)
					pb[loop] = (double)point_A[loop];
				parcial.clear();
			}
			else
				parcial.clear();
		}
		// int num_inliers = best.size();
		// return num_inliers;
	}

	void getPDF(std::vector<double> &charge, double Tcharge, int size, std::vector<double> &PDF)
	{
		int loop;
		for (loop = 0; loop < size; loop++)
			PDF[loop] = (double)charge[loop] / Tcharge;
	}

	void CrossProd(double *C, double *A, double *B)
	{
		C[0] = (double)A[1] * B[2] - A[2] * B[1];
		C[1] = (double)-(A[0] * B[2] - A[2] * B[0]);
		C[2] = (double)A[0] * B[1] - A[1] * B[0];
	}

	void CrossProd2(double *C, std::vector<double> A, double *B)
	{
		C[0] = (double)A[1] * B[2] - A[2] * B[1];
		C[1] = (double)-(A[0] * B[2] - A[2] * B[0]);
		C[2] = (double)A[0] * B[1] - A[1] * B[0];
	}

	double Norma(double *A)
	{
		return (double)sqrt((double)pow(A[0], 2.) + (double)pow(A[1], 2.) + (double)pow(A[2], 2.));
	}

	void argsort(std::vector<double> &nums, std::vector<int> &indices)
	{
		// int n = nums.size();
		// std::vector<int> indices(n);
		std::iota(indices.begin(), indices.end(), 0);
		sort(indices.begin(), indices.end(), [&nums](int i, int j)
			 { return nums[i] < nums[j]; });
		// return indices;
	}

	void get_random2(int &ind1, int &ind2, std::vector<std::vector<double>> &data, std::vector<double> &charge, std::vector<double> &PDF, double TCharge, double AvgCharge, double TwiceAvCharge, int mode, int size)
	{
		if (mode == 0)
		{
			// Random sampling
			ind1 = rand() % size;
			ind2 = rand() % size;
			while (ind1 == ind2)
				ind2 = rand() % size;
		}
		else if (mode == 1)
		{
			// Gaussian sampling
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<> dis(0.0, 1.0);
			int loop;
			double dist = 0;
			double sigma = 30.0;
			double y = 0;
			double gauss = 0;
			int counter = 0;
			double dif[3];
			double P2[3];
			double P1[3];
			ind1 = rand() % size;
			do
			{
				ind2 = rand() % size;
				for (loop = 0; loop < 3; loop++)
					P1[loop] = data[ind1][loop];
				for (loop = 0; loop < 3; loop++)
					P2[loop] = data[ind2][loop];
				for (loop = 0; loop < 3; loop++)
					dif[loop] = P2[loop] - P1[loop];
				dist = sqrt(pow(dif[0], 2.0) + pow(dif[1], 2.0) + pow(dif[2], 2.0));
				gauss = 1.0 * exp(-1.0 * pow(dist / sigma, 2.0));
				y = dis(gen);
				counter++;
				if (counter > 20 && ind2 != ind1)
					break;
			} while (ind1 == ind2 || y > gauss);
		}
		else if (mode == 2)
		{
			// Weighted sampling

			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<> dis(0.0, 1.0);

			bool cond = false;
			int counter = 0;
			double w2;

			ind1 = rand() % size;
			do
			{
				counter++;
				if (counter > 30 && ind2 != ind1)
					break;
				ind2 = rand() % size;
				cond = false;
				w2 = dis(gen) * TwiceAvCharge;
				if (PDF[ind2] >= w2)
					cond = true;
			} while (ind2 == ind1 || cond == false);
		}

		else if (mode == 3)
		{
			// Weighted sampling + Gauss dist.
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<> dis(0.0, 1.0);
			bool cond = false;
			double dist = 0;
			double sigma = 30.0;
			double y = 0;
			double gauss = 0;
			int counter = 0;
			double dif[3];
			double P2[3];
			double P1[3];
			double w2;
			int loop;
			ind1 = rand() % size;
			do
			{
				ind2 = rand() % size;
				for (loop = 0; loop < 3; loop++)
				{
					P1[loop] = data[ind1][loop];
					P2[loop] = data[ind2][loop];
					dif[loop] = P2[loop] - P1[loop];
				}
				dist = sqrt(pow(dif[0], 2.0) + pow(dif[1], 2.0) + pow(dif[2], 2.0));
				gauss = 1.0 * exp(-1.0 * pow(dist / sigma, 2));
				y = dis(gen);
				counter++;
				if (counter > 20 && ind2 != ind1)
					break;
				cond = false;
				w2 = dis(gen) * TwiceAvCharge;
				if (PDF[ind2] >= w2)
					cond = true;
				else
				{
					w2 = 1.0;
					cond = true;
				}
			} while (ind2 == ind1 || cond == false || y > gauss);
		}
	}

	void Fit3D_i(std::vector<std::vector<double>> &data, std::vector<int> &inliers, std::vector<double> &versor, std::vector<double> &Pb)
	{
		// 3D Line Regression
		// adapted from: https://github.com/jczamorac/Tracking_RANSAC/

		// long double PI = 3.14159265358979323851;

		// int R, C;
		const int size = (const int)inliers.size();
		double Q;
		double Xm, Ym, Zm;
		double Xh, Yh, Zh;
		double a, b;
		double Sxx, Sxy, Syy, Sxz, Szz, Syz;
		double theta;
		double K11, K22, K12, K10, K01, K00;
		double c0, c1, c2;
		double p, q, r, dm2;
		double rho, phi;
		int i;

		std::vector<double> vX(size), vY(size), vZ(size), vQ(size);

		for (int i = 0; i < size; i++)
		{
			vX[i] = data[inliers[i]][0];
			vY[i] = data[inliers[i]][1];
			vZ[i] = data[inliers[i]][2];
			vQ[i] = data[inliers[i]][3];
		}

		Q = Xm = Ym = Zm = 0.;
		double total_charge = 0;
		Sxx = Syy = Szz = Sxy = Sxz = Syz = 0.;

		for (i = 0; i < size; i++)
		{
			Q += vQ[i] / 10.;
			Xm += vX[i] * vQ[i] / 10.;
			Ym += vY[i] * vQ[i] / 10.;
			Zm += vZ[i] * vQ[i] / 10.;
			Sxx += vX[i] * vX[i] * vQ[i] / 10.;
			Syy += vY[i] * vY[i] * vQ[i] / 10.;
			Szz += vZ[i] * vZ[i] * vQ[i] / 10.;
			Sxy += vX[i] * vY[i] * vQ[i] / 10.;
			Sxz += vX[i] * vZ[i] * vQ[i] / 10.;
			Syz += vY[i] * vZ[i] * vQ[i] / 10.;
		}

		Xm /= Q;
		Ym /= Q;
		Zm /= Q;
		Sxx /= Q;
		Syy /= Q;
		Szz /= Q;
		Sxy /= Q;
		Sxz /= Q;
		Syz /= Q;
		Sxx -= (Xm * Xm);
		Syy -= (Ym * Ym);
		Szz -= (Zm * Zm);
		Sxy -= (Xm * Ym);
		Sxz -= (Xm * Zm);
		Syz -= (Ym * Zm);

		theta = 0.5 * atan((2. * Sxy) / (Sxx - Syy));

		K11 = (Syy + Szz) * pow(cos(theta), 2) + (Sxx + Szz) * pow(sin(theta), 2) - 2. * Sxy * cos(theta) * sin(theta);
		K22 = (Syy + Szz) * pow(sin(theta), 2) + (Sxx + Szz) * pow(cos(theta), 2) + 2. * Sxy * cos(theta) * sin(theta);
		K12 = -Sxy * (pow(cos(theta), 2) - pow(sin(theta), 2)) + (Sxx - Syy) * cos(theta) * sin(theta);
		K10 = Sxz * cos(theta) + Syz * sin(theta);
		K01 = -Sxz * sin(theta) + Syz * cos(theta);
		K00 = Sxx + Syy;

		c2 = -K00 - K11 - K22;
		c1 = K00 * K11 + K00 * K22 + K11 * K22 - K01 * K01 - K10 * K10;
		c0 = K01 * K01 * K11 + K10 * K10 * K22 - K00 * K11 * K22;

		p = c1 - pow(c2, 2) / 3.;
		q = 2. * pow(c2, 3) / 27. - c1 * c2 / 3. + c0;
		r = pow(q / 2., 2) + pow(p, 3) / 27.;

		if (r > 0)
			dm2 = -c2 / 3. + pow(-q / 2. + sqrt(r), 1. / 3.) + pow(-q / 2. - sqrt(r), 1. / 3.);
		if (r < 0)
		{
			rho = sqrt(-pow(p, 3) / 27.);
			phi = acos(-q / (2. * rho));
			dm2 = std::min(-c2 / 3. + 2. * pow(rho, 1. / 3.) * cos(phi / 3.), std::min(-c2 / 3. + 2. * pow(rho, 1. / 3.) * cos((phi + 2. * M_PI) / 3.), -c2 / 3. + 2. * pow(rho, 1. / 3.) * cos((phi + 4. * M_PI) / 3.)));
		}

		a = -K10 * cos(theta) / (K11 - dm2) + K01 * sin(theta) / (K22 - dm2);
		b = -K10 * sin(theta) / (K11 - dm2) - K01 * cos(theta) / (K22 - dm2);

		Xh = ((1. + b * b) * Xm - a * b * Ym + a * Zm) / (1. + a * a + b * b);
		Yh = ((1. + a * a) * Ym - a * b * Xm + b * Zm) / (1. + a * a + b * b);
		Zh = ((a * a + b * b) * Zm + a * Xm + b * Ym) / (1. + a * a + b * b);

		versor[0] = (double)Xh - Xm;
		versor[1] = (double)Yh - Ym;
		versor[2] = (double)Zh - Zm;

		double norma = sqrt(pow(versor[0], 2.) + pow(versor[1], 2.) + pow(versor[2], 2.));

		for (i = 0; i < 3; i++)
			versor[i] = (double)versor[i] / norma;

		Pb[0] = Xm;
		Pb[1] = Ym;
		Pb[2] = Zm;
	}

	void Fit3D2(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &vZ, std::vector<double> &vQ, std::vector<double> &versor, std::vector<double> &Pb)
	{
		// 3D Line Regression
		// adapted from: https://github.com/jczamorac/Tracking_RANSAC/

		// long double PI = 3.14159265358979323851;

		// int R, C;
		double Q;
		double Xm, Ym, Zm;
		double Xh, Yh, Zh;
		double a, b;
		double Sxx, Sxy, Syy, Sxz, Szz, Syz;
		double theta;
		double K11, K22, K12, K10, K01, K00;
		double c0, c1, c2;
		double p, q, r, dm2;
		double rho, phi;
		int i, size = vX.size();

		Q = Xm = Ym = Zm = 0.;
		double total_charge = 0;
		Sxx = Syy = Szz = Sxy = Sxz = Syz = 0.;

		for (i = 0; i < size; i++)
		{
			Q += vQ[i] / 10.;
			Xm += vX[i] * vQ[i] / 10.;
			Ym += vY[i] * vQ[i] / 10.;
			Zm += vZ[i] * vQ[i] / 10.;
			Sxx += vX[i] * vX[i] * vQ[i] / 10.;
			Syy += vY[i] * vY[i] * vQ[i] / 10.;
			Szz += vZ[i] * vZ[i] * vQ[i] / 10.;
			Sxy += vX[i] * vY[i] * vQ[i] / 10.;
			Sxz += vX[i] * vZ[i] * vQ[i] / 10.;
			Syz += vY[i] * vZ[i] * vQ[i] / 10.;
		}

		Xm /= Q;
		Ym /= Q;
		Zm /= Q;
		Sxx /= Q;
		Syy /= Q;
		Szz /= Q;
		Sxy /= Q;
		Sxz /= Q;
		Syz /= Q;
		Sxx -= (Xm * Xm);
		Syy -= (Ym * Ym);
		Szz -= (Zm * Zm);
		Sxy -= (Xm * Ym);
		Sxz -= (Xm * Zm);
		Syz -= (Ym * Zm);

		theta = 0.5 * atan((2. * Sxy) / (Sxx - Syy));

		K11 = (Syy + Szz) * pow(cos(theta), 2) + (Sxx + Szz) * pow(sin(theta), 2) - 2. * Sxy * cos(theta) * sin(theta);
		K22 = (Syy + Szz) * pow(sin(theta), 2) + (Sxx + Szz) * pow(cos(theta), 2) + 2. * Sxy * cos(theta) * sin(theta);
		K12 = -Sxy * (pow(cos(theta), 2) - pow(sin(theta), 2)) + (Sxx - Syy) * cos(theta) * sin(theta);
		K10 = Sxz * cos(theta) + Syz * sin(theta);
		K01 = -Sxz * sin(theta) + Syz * cos(theta);
		K00 = Sxx + Syy;

		c2 = -K00 - K11 - K22;
		c1 = K00 * K11 + K00 * K22 + K11 * K22 - K01 * K01 - K10 * K10;
		c0 = K01 * K01 * K11 + K10 * K10 * K22 - K00 * K11 * K22;

		p = c1 - pow(c2, 2) / 3.;
		q = 2. * pow(c2, 3) / 27. - c1 * c2 / 3. + c0;
		r = pow(q / 2., 2) + pow(p, 3) / 27.;

		if (r > 0)
			dm2 = -c2 / 3. + pow(-q / 2. + sqrt(r), 1. / 3.) + pow(-q / 2. - sqrt(r), 1. / 3.);
		if (r < 0)
		{
			rho = sqrt(-pow(p, 3) / 27.);
			phi = acos(-q / (2. * rho));
			dm2 = std::min(-c2 / 3. + 2. * pow(rho, 1. / 3.) * cos(phi / 3.), std::min(-c2 / 3. + 2. * pow(rho, 1. / 3.) * cos((phi + 2. * M_PI) / 3.), -c2 / 3. + 2. * pow(rho, 1. / 3.) * cos((phi + 4. * M_PI) / 3.)));
		}

		a = -K10 * cos(theta) / (K11 - dm2) + K01 * sin(theta) / (K22 - dm2);
		b = -K10 * sin(theta) / (K11 - dm2) - K01 * cos(theta) / (K22 - dm2);

		Xh = ((1. + b * b) * Xm - a * b * Ym + a * Zm) / (1. + a * a + b * b);
		Yh = ((1. + a * a) * Ym - a * b * Xm + b * Zm) / (1. + a * a + b * b);
		Zh = ((a * a + b * b) * Zm + a * Xm + b * Ym) / (1. + a * a + b * b);

		versor[0] = (double)Xh - Xm;
		versor[1] = (double)Yh - Ym;
		versor[2] = (double)Zh - Zm;

		double norma = sqrt(pow(versor[0], 2.) + pow(versor[1], 2.) + pow(versor[2], 2.));

		for (i = 0; i < 3; i++)
			versor[i] = (double)versor[i] / norma;

		Pb[0] = Xm;
		Pb[1] = Ym;
		Pb[2] = Zm;
	}

	void pRansac(std::vector<std::vector<double>> &data, PyObject *PyInliers, PyObject *PyVersors, PyObject *PyPoints, std::vector<double> &charge, int number_it, double min_dist, int mode, int min_inlier)
	{
		int indice1, indice2, loop, j, size = data.size(), num_inliers_1 = 0;
		std::vector<int> parcial;
		std::vector<std::vector<double>> versores;
		std::vector<std::vector<double>> points;
		std::vector<double> pesos;
		double parcial_versor[3];
		double norma1, norma2;
		double point_A[3];
		double point_B[3];
		double aux[3];
		double cross[3];
		double TCharge = 0;
		std::vector<double> PDF(size);
		double AvgCharge, TwiceAvCharge;
		double distancias_quadrado = 0.;
		parcial.reserve(size);
		versores.reserve(int(number_it));
		points.reserve(int(number_it));

		if (mode == 2 || mode == 3)
		{
			for (loop = 0; loop < size; loop++)
				TCharge += charge[loop];
			AvgCharge = (double)TCharge / size;
			TwiceAvCharge = (double)2 * AvgCharge;
			getPDF(charge, TCharge, size, PDF);
		}

		for (int i = 0; i < number_it; i++)
		{

			get_random2(indice1, indice2, data, charge, PDF, TCharge, AvgCharge, TwiceAvCharge, mode, size);

			for (loop = 0; loop < 3; loop++)
			{
				point_A[loop] = data[indice1][loop];
				point_B[loop] = data[indice2][loop];
				parcial_versor[loop] = (double)point_B[loop] - point_A[loop];
			}

			norma1 = Norma(parcial_versor);
			for (loop = 0; loop < 3; loop++)
				parcial_versor[loop] = (double)parcial_versor[loop] / norma1;
			for (j = 0; j < size; j++)
			{
				for (loop = 0; loop < 3; loop++)
					aux[loop] = point_A[loop] - data[j][loop];

				CrossProd(cross, parcial_versor, aux);
				norma2 = Norma(cross);

				if (fabs(norma2) <= min_dist)
				{
					// parcial.push_back(j);
					num_inliers_1 += 1;
					distancias_quadrado += (double)pow(norma2, 2.0);
				}
			}
			// double p_weight = (double) distancias_quadrado/parcial.size();
			double p_weight = (double)distancias_quadrado / num_inliers_1;
			// if (parcial.size() >= min_inlier){
			if (num_inliers_1 >= min_inlier)
			{
				pesos.push_back(p_weight);
				std::vector<double> new_versor{parcial_versor[0], parcial_versor[1], parcial_versor[2]};
				std::vector<double> new_point{point_A[0], point_A[1], point_A[2]};
				versores.push_back(new_versor);
				points.push_back(new_point);
			}

			distancias_quadrado = 0.0;
			// parcial.clear();
			num_inliers_1 = 0;
		}
		std::vector<int> args(pesos.size());
		argsort(pesos, args);
		std::vector<int> indices(size);
		// std::cout << pesos.size() << " " << args.size() << "\n";
		for (int z = 0; z < size; z++)
			indices[z] = z;

		double norma, versor[3], p[3];
		int i, k, l, m;

		for (i = 0; i < pesos.size(); i++)
		{

			for (j = 0; j < 3; j++)
			{
				versor[j] = versores[args[i]][j];
				p[j] = points[args[i]][j];
			}
#pragma omp parallel for
			for (k = 0; k < indices.size(); k++)
			{

				for (l = 0; l < 3; l++)
					aux[l] = p[l] - data[indices[k]][l];

				CrossProd(cross, versor, aux);
				norma = Norma(cross);

				if (fabs(norma) <= min_dist)
					parcial.push_back(indices[k]);
			}

			if (parcial.size() >= min_inlier)
			{

				// PyObject* parcial_inliers = PyList_New(parcial.size());
				PyObject *parcial_V = PyList_New(3);
				PyObject *parcial_P = PyList_New(3);

				// Adaptação para pegar apenas quais os inliers
				PyObject *int_vector = PyList_New(parcial.size());
				for (m = 0; m < parcial.size(); m++)
					PyList_SetItem(int_vector, m, PyLong_FromLong((long int)parcial[m]));
				PyList_Append(PyInliers, PyArray_FROM_OTF(int_vector, NPY_INT, NPY_ARRAY_IN_ARRAY));
				std::vector<double> versor22(3);
				std::vector<double> p22(3);
				Fit3D_i(data, parcial, versor22, p22);
				std::vector<int> tempRemain;
				std::set_difference(indices.begin(), indices.end(), parcial.begin(), parcial.end(),
									std::inserter(tempRemain, tempRemain.begin()));
				indices = tempRemain;
				tempRemain.clear();

				// Guarda versor e ponto
				PyList_SetItem(parcial_V, 0, PyFloat_FromDouble(versor22[0]));
				PyList_SetItem(parcial_V, 1, PyFloat_FromDouble(versor22[1]));
				PyList_SetItem(parcial_V, 2, PyFloat_FromDouble(versor22[2]));
				PyList_SetItem(parcial_P, 0, PyFloat_FromDouble(p22[0]));
				PyList_SetItem(parcial_P, 1, PyFloat_FromDouble(p22[1]));
				PyList_SetItem(parcial_P, 2, PyFloat_FromDouble(p22[2]));
				PyList_Append(PyVersors, PyArray_FROM_OTF(parcial_V, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
				PyList_Append(PyPoints, PyArray_FROM_OTF(parcial_P, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
			}
			parcial.clear();
			if (indices.size() < min_inlier)
				break;
		}
	}

	int Background_root(double *spectrum, int ssize, int numberIterations, int direction, int filterOrder, bool smoothing, int smoothWindow, bool compton)
	{
		int i, j, w, bw, b1, b2, priz;
		double a, b, c, d, e, yb1, yb2, ai, av, men, b4, c4, d4, e4, b6, c6, d6, e6, f6, g6, b8, c8, d8, e8, f8, g8, h8, i8;
		if (ssize < 2 * numberIterations + 1)
		{
			// PyErr_SetString(PyExc_TypeError, "Too Large Clipping Window.");
			return -1;
		}
		if (smoothing == true && smoothWindow != 3 && smoothWindow != 5 && smoothWindow != 7 && smoothWindow != 9 && smoothWindow != 11 && smoothWindow != 13 && smoothWindow != 15)
		{
			// PyErr_SetString(PyExc_TypeError, "Incorrect width of smoothing window");
			return -2;
		}
		double *working_space = new double[2 * ssize];
		for (i = 0; i < ssize; i++)
		{
			working_space[i] = spectrum[i];
			working_space[i + ssize] = spectrum[i];
		}
		bw = (smoothWindow - 1) / 2;
		if (direction == 0)
			i = 1;
		else if (direction == 1)
			i = numberIterations;
		if (filterOrder == 0)
		{
			do
			{
				for (j = i; j < ssize - i; j++)
				{
					if (smoothing == false)
					{
						a = working_space[ssize + j];
						b = (working_space[ssize + j - i] + working_space[ssize + j + i]) / 2.0;
						if (b < a)
							a = b;
						working_space[j] = a;
					}

					else if (smoothing == true)
					{
						a = working_space[ssize + j];
						av = 0;
						men = 0;
						for (w = j - bw; w <= j + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								av += working_space[ssize + w];
								men += 1;
							}
						}
						av = av / men;
						b = 0;
						men = 0;
						for (w = j - i - bw; w <= j - i + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								b += working_space[ssize + w];
								men += 1;
							}
						}
						b = b / men;
						c = 0;
						men = 0;
						for (w = j + i - bw; w <= j + i + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								c += working_space[ssize + w];
								men += 1;
							}
						}
						c = c / men;
						b = (b + c) / 2;
						if (b < a)
							av = b;
						working_space[j] = av;
					}
				}
				for (j = i; j < ssize - i; j++)
					working_space[ssize + j] = working_space[j];
				if (direction == 0)
					i += 1;
				else if (direction == 1)
					i -= 1;
			} while ((direction == 0 && i <= numberIterations) || (direction == 1 && i >= 1));
		}

		else if (filterOrder == 1)
		{
			do
			{
				for (j = i; j < ssize - i; j++)
				{
					if (smoothing == false)
					{
						a = working_space[ssize + j];
						b = (working_space[ssize + j - i] + working_space[ssize + j + i]) / 2.0;
						c = 0;
						ai = i / 2;
						c -= working_space[ssize + j - (int)(2 * ai)] / 6;
						c += 4 * working_space[ssize + j - (int)ai] / 6;
						c += 4 * working_space[ssize + j + (int)ai] / 6;
						c -= working_space[ssize + j + (int)(2 * ai)] / 6;
						if (b < c)
							b = c;
						if (b < a)
							a = b;
						working_space[j] = a;
					}

					else if (smoothing == true)
					{
						a = working_space[ssize + j];
						av = 0;
						men = 0;
						for (w = j - bw; w <= j + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								av += working_space[ssize + w];
								men += 1;
							}
						}
						av = av / men;
						b = 0;
						men = 0;
						for (w = j - i - bw; w <= j - i + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								b += working_space[ssize + w];
								men += 1;
							}
						}
						b = b / men;
						c = 0;
						men = 0;
						for (w = j + i - bw; w <= j + i + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								c += working_space[ssize + w];
								men += 1;
							}
						}
						c = c / men;
						b = (b + c) / 2;
						ai = i / 2;
						b4 = 0, men = 0;
						for (w = j - (int)(2 * ai) - bw; w <= j - (int)(2 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								b4 += working_space[ssize + w];
								men += 1;
							}
						}
						b4 = b4 / men;
						c4 = 0, men = 0;
						for (w = j - (int)ai - bw; w <= j - (int)ai + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								c4 += working_space[ssize + w];
								men += 1;
							}
						}
						c4 = c4 / men;
						d4 = 0, men = 0;
						for (w = j + (int)ai - bw; w <= j + (int)ai + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								d4 += working_space[ssize + w];
								men += 1;
							}
						}
						d4 = d4 / men;
						e4 = 0, men = 0;
						for (w = j + (int)(2 * ai) - bw; w <= j + (int)(2 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								e4 += working_space[ssize + w];
								men += 1;
							}
						}
						e4 = e4 / men;
						b4 = (-b4 + 4 * c4 + 4 * d4 - e4) / 6;
						if (b < b4)
							b = b4;
						if (b < a)
							av = b;
						working_space[j] = av;
					}
				}
				for (j = i; j < ssize - i; j++)
					working_space[ssize + j] = working_space[j];
				if (direction == 0)
					i += 1;
				else if (direction == 1)
					i -= 1;
			} while ((direction == 0 && i <= numberIterations) || (direction == 1 && i >= 1));
		}

		else if (filterOrder == 2)
		{
			do
			{
				for (j = i; j < ssize - i; j++)
				{
					if (smoothing == false)
					{
						a = working_space[ssize + j];
						b = (working_space[ssize + j - i] + working_space[ssize + j + i]) / 2.0;
						c = 0;
						ai = i / 2;
						c -= working_space[ssize + j - (int)(2 * ai)] / 6;
						c += 4 * working_space[ssize + j - (int)ai] / 6;
						c += 4 * working_space[ssize + j + (int)ai] / 6;
						c -= working_space[ssize + j + (int)(2 * ai)] / 6;
						d = 0;
						ai = i / 3;
						d += working_space[ssize + j - (int)(3 * ai)] / 20;
						d -= 6 * working_space[ssize + j - (int)(2 * ai)] / 20;
						d += 15 * working_space[ssize + j - (int)ai] / 20;
						d += 15 * working_space[ssize + j + (int)ai] / 20;
						d -= 6 * working_space[ssize + j + (int)(2 * ai)] / 20;
						d += working_space[ssize + j + (int)(3 * ai)] / 20;
						if (b < d)
							b = d;
						if (b < c)
							b = c;
						if (b < a)
							a = b;
						working_space[j] = a;
					}

					else if (smoothing == true)
					{
						a = working_space[ssize + j];
						av = 0;
						men = 0;
						for (w = j - bw; w <= j + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								av += working_space[ssize + w];
								men += 1;
							}
						}
						av = av / men;
						b = 0;
						men = 0;
						for (w = j - i - bw; w <= j - i + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								b += working_space[ssize + w];
								men += 1;
							}
						}
						b = b / men;
						c = 0;
						men = 0;
						for (w = j + i - bw; w <= j + i + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								c += working_space[ssize + w];
								men += 1;
							}
						}
						c = c / men;
						b = (b + c) / 2;
						ai = i / 2;
						b4 = 0, men = 0;
						for (w = j - (int)(2 * ai) - bw; w <= j - (int)(2 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								b4 += working_space[ssize + w];
								men += 1;
							}
						}
						b4 = b4 / men;
						c4 = 0, men = 0;
						for (w = j - (int)ai - bw; w <= j - (int)ai + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								c4 += working_space[ssize + w];
								men += 1;
							}
						}
						c4 = c4 / men;
						d4 = 0, men = 0;
						for (w = j + (int)ai - bw; w <= j + (int)ai + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								d4 += working_space[ssize + w];
								men += 1;
							}
						}
						d4 = d4 / men;
						e4 = 0, men = 0;
						for (w = j + (int)(2 * ai) - bw; w <= j + (int)(2 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								e4 += working_space[ssize + w];
								men += 1;
							}
						}
						e4 = e4 / men;
						b4 = (-b4 + 4 * c4 + 4 * d4 - e4) / 6;
						ai = i / 3;
						b6 = 0, men = 0;
						for (w = j - (int)(3 * ai) - bw; w <= j - (int)(3 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								b6 += working_space[ssize + w];
								men += 1;
							}
						}
						b6 = b6 / men;
						c6 = 0, men = 0;
						for (w = j - (int)(2 * ai) - bw; w <= j - (int)(2 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								c6 += working_space[ssize + w];
								men += 1;
							}
						}
						c6 = c6 / men;
						d6 = 0, men = 0;
						for (w = j - (int)ai - bw; w <= j - (int)ai + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								d6 += working_space[ssize + w];
								men += 1;
							}
						}
						d6 = d6 / men;
						e6 = 0, men = 0;
						for (w = j + (int)ai - bw; w <= j + (int)ai + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								e6 += working_space[ssize + w];
								men += 1;
							}
						}
						e6 = e6 / men;
						f6 = 0, men = 0;
						for (w = j + (int)(2 * ai) - bw; w <= j + (int)(2 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								f6 += working_space[ssize + w];
								men += 1;
							}
						}
						f6 = f6 / men;
						g6 = 0, men = 0;
						for (w = j + (int)(3 * ai) - bw; w <= j + (int)(3 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								g6 += working_space[ssize + w];
								men += 1;
							}
						}
						g6 = g6 / men;
						b6 = (b6 - 6 * c6 + 15 * d6 + 15 * e6 - 6 * f6 + g6) / 20;
						if (b < b6)
							b = b6;
						if (b < b4)
							b = b4;
						if (b < a)
							av = b;
						working_space[j] = av;
					}
				}
				for (j = i; j < ssize - i; j++)
					working_space[ssize + j] = working_space[j];
				if (direction == 0)
					i += 1;
				else if (direction == 1)
					i -= 1;
			} while ((direction == 0 && i <= numberIterations) || (direction == 1 && i >= 1));
		}

		else if (filterOrder == 3)
		{
			do
			{
				for (j = i; j < ssize - i; j++)
				{
					if (smoothing == false)
					{
						a = working_space[ssize + j];
						b = (working_space[ssize + j - i] + working_space[ssize + j + i]) / 2.0;
						c = 0;
						ai = i / 2;
						c -= working_space[ssize + j - (int)(2 * ai)] / 6;
						c += 4 * working_space[ssize + j - (int)ai] / 6;
						c += 4 * working_space[ssize + j + (int)ai] / 6;
						c -= working_space[ssize + j + (int)(2 * ai)] / 6;
						d = 0;
						ai = i / 3;
						d += working_space[ssize + j - (int)(3 * ai)] / 20;
						d -= 6 * working_space[ssize + j - (int)(2 * ai)] / 20;
						d += 15 * working_space[ssize + j - (int)ai] / 20;
						d += 15 * working_space[ssize + j + (int)ai] / 20;
						d -= 6 * working_space[ssize + j + (int)(2 * ai)] / 20;
						d += working_space[ssize + j + (int)(3 * ai)] / 20;
						e = 0;
						ai = i / 4;
						e -= working_space[ssize + j - (int)(4 * ai)] / 70;
						e += 8 * working_space[ssize + j - (int)(3 * ai)] / 70;
						e -= 28 * working_space[ssize + j - (int)(2 * ai)] / 70;
						e += 56 * working_space[ssize + j - (int)ai] / 70;
						e += 56 * working_space[ssize + j + (int)ai] / 70;
						e -= 28 * working_space[ssize + j + (int)(2 * ai)] / 70;
						e += 8 * working_space[ssize + j + (int)(3 * ai)] / 70;
						e -= working_space[ssize + j + (int)(4 * ai)] / 70;
						if (b < e)
							b = e;
						if (b < d)
							b = d;
						if (b < c)
							b = c;
						if (b < a)
							a = b;
						working_space[j] = a;
					}

					else if (smoothing == true)
					{
						a = working_space[ssize + j];
						av = 0;
						men = 0;
						for (w = j - bw; w <= j + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								av += working_space[ssize + w];
								men += 1;
							}
						}
						av = av / men;
						b = 0;
						men = 0;
						for (w = j - i - bw; w <= j - i + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								b += working_space[ssize + w];
								men += 1;
							}
						}
						b = b / men;
						c = 0;
						men = 0;
						for (w = j + i - bw; w <= j + i + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								c += working_space[ssize + w];
								men += 1;
							}
						}
						c = c / men;
						b = (b + c) / 2;
						ai = i / 2;
						b4 = 0, men = 0;
						for (w = j - (int)(2 * ai) - bw; w <= j - (int)(2 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								b4 += working_space[ssize + w];
								men += 1;
							}
						}
						b4 = b4 / men;
						c4 = 0, men = 0;
						for (w = j - (int)ai - bw; w <= j - (int)ai + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								c4 += working_space[ssize + w];
								men += 1;
							}
						}
						c4 = c4 / men;
						d4 = 0, men = 0;
						for (w = j + (int)ai - bw; w <= j + (int)ai + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								d4 += working_space[ssize + w];
								men += 1;
							}
						}
						d4 = d4 / men;
						e4 = 0, men = 0;
						for (w = j + (int)(2 * ai) - bw; w <= j + (int)(2 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								e4 += working_space[ssize + w];
								men += 1;
							}
						}
						e4 = e4 / men;
						b4 = (-b4 + 4 * c4 + 4 * d4 - e4) / 6;
						ai = i / 3;
						b6 = 0, men = 0;
						for (w = j - (int)(3 * ai) - bw; w <= j - (int)(3 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								b6 += working_space[ssize + w];
								men += 1;
							}
						}
						b6 = b6 / men;
						c6 = 0, men = 0;
						for (w = j - (int)(2 * ai) - bw; w <= j - (int)(2 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								c6 += working_space[ssize + w];
								men += 1;
							}
						}
						c6 = c6 / men;
						d6 = 0, men = 0;
						for (w = j - (int)ai - bw; w <= j - (int)ai + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								d6 += working_space[ssize + w];
								men += 1;
							}
						}
						d6 = d6 / men;
						e6 = 0, men = 0;
						for (w = j + (int)ai - bw; w <= j + (int)ai + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								e6 += working_space[ssize + w];
								men += 1;
							}
						}
						e6 = e6 / men;
						f6 = 0, men = 0;
						for (w = j + (int)(2 * ai) - bw; w <= j + (int)(2 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								f6 += working_space[ssize + w];
								men += 1;
							}
						}
						f6 = f6 / men;
						g6 = 0, men = 0;
						for (w = j + (int)(3 * ai) - bw; w <= j + (int)(3 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								g6 += working_space[ssize + w];
								men += 1;
							}
						}
						g6 = g6 / men;
						b6 = (b6 - 6 * c6 + 15 * d6 + 15 * e6 - 6 * f6 + g6) / 20;
						ai = i / 4;
						b8 = 0, men = 0;
						for (w = j - (int)(4 * ai) - bw; w <= j - (int)(4 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								b8 += working_space[ssize + w];
								men += 1;
							}
						}
						b8 = b8 / men;
						c8 = 0, men = 0;
						for (w = j - (int)(3 * ai) - bw; w <= j - (int)(3 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								c8 += working_space[ssize + w];
								men += 1;
							}
						}
						c8 = c8 / men;
						d8 = 0, men = 0;
						for (w = j - (int)(2 * ai) - bw; w <= j - (int)(2 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								d8 += working_space[ssize + w];
								men += 1;
							}
						}
						d8 = d8 / men;
						e8 = 0, men = 0;
						for (w = j - (int)ai - bw; w <= j - (int)ai + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								e8 += working_space[ssize + w];
								men += 1;
							}
						}
						e8 = e8 / men;
						f8 = 0, men = 0;
						for (w = j + (int)ai - bw; w <= j + (int)ai + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								f8 += working_space[ssize + w];
								men += 1;
							}
						}
						f8 = f8 / men;
						g8 = 0, men = 0;
						for (w = j + (int)(2 * ai) - bw; w <= j + (int)(2 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								g8 += working_space[ssize + w];
								men += 1;
							}
						}
						g8 = g8 / men;
						h8 = 0, men = 0;
						for (w = j + (int)(3 * ai) - bw; w <= j + (int)(3 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								h8 += working_space[ssize + w];
								men += 1;
							}
						}
						h8 = h8 / men;
						i8 = 0, men = 0;
						for (w = j + (int)(4 * ai) - bw; w <= j + (int)(4 * ai) + bw; w++)
						{
							if (w >= 0 && w < ssize)
							{
								i8 += working_space[ssize + w];
								men += 1;
							}
						}
						i8 = i8 / men;
						b8 = (-b8 + 8 * c8 - 28 * d8 + 56 * e8 - 56 * f8 - 28 * g8 + 8 * h8 - i8) / 70;
						if (b < b8)
							b = b8;
						if (b < b6)
							b = b6;
						if (b < b4)
							b = b4;
						if (b < a)
							av = b;
						working_space[j] = av;
					}
				}
				for (j = i; j < ssize - i; j++)
					working_space[ssize + j] = working_space[j];
				if (direction == 0)
					i += 1;
				else if (direction == 1)
					i -= 1;
			} while ((direction == 0 && i <= numberIterations) || (direction == 1 && i >= 1));
		}

		if (compton == true)
		{
			for (i = 0, b2 = 0; i < ssize; i++)
			{
				b1 = b2;
				a = working_space[i], b = spectrum[i];
				j = i;
				if (fabs(a - b) >= 1)
				{
					b1 = i - 1;
					if (b1 < 0)
						b1 = 0;
					yb1 = working_space[b1];
					for (b2 = b1 + 1, c = 0, priz = 0; priz == 0 && b2 < ssize; b2++)
					{
						a = working_space[b2], b = spectrum[b2];
						c = c + b - yb1;
						if (fabs(a - b) < 1)
						{
							priz = 1;
							yb2 = b;
						}
					}
					if (b2 == ssize)
						b2 -= 1;
					yb2 = working_space[b2];
					if (yb1 <= yb2)
					{
						for (j = b1, c = 0; j <= b2; j++)
						{
							b = spectrum[j];
							c = c + b - yb1;
						}
						if (c > 1)
						{
							c = (yb2 - yb1) / c;
							for (j = b1, d = 0; j <= b2 && j < ssize; j++)
							{
								b = spectrum[j];
								d = d + b - yb1;
								a = c * d + yb1;
								working_space[ssize + j] = a;
							}
						}
					}

					else
					{
						for (j = b2, c = 0; j >= b1; j--)
						{
							b = spectrum[j];
							c = c + b - yb2;
						}
						if (c > 1)
						{
							c = (yb1 - yb2) / c;
							for (j = b2, d = 0; j >= b1 && j >= 0; j--)
							{
								b = spectrum[j];
								d = d + b - yb2;
								a = c * d + yb2;
								working_space[ssize + j] = a;
							}
						}
					}
					i = b2;
				}
			}
		}

		for (j = 0; j < ssize; j++)
			spectrum[j] = working_space[ssize + j];
		delete[] working_space;
		return 0;
	}

	int SearchHighRes_root(double *source, double *destVector, double *fPositionX, int ssize, double sigma, double threshold, bool backgroundRemove, int deconIterations, bool markov, int averWindow)
	{
		int i, j, numberIterations = (int)(7 * sigma + 0.5);
		double a, b, c;
		int k, lindex, posit, imin, imax, jmin, jmax, lh_gold, priz, fMaxPeaks = ssize;
		double lda, ldb, ldc, area, maximum, maximum_decon;
		int xmin, xmax, l, peak_index = 0, size_ext = ssize + 2 * numberIterations, shift = numberIterations, bw = 2, w;
		double maxch;
		double nom, nip, nim, sp, sm, plocha = 0;
		double m0low = 0, m1low = 0, m2low = 0, l0low = 0, l1low = 0, detlow, av, men;
		// double *fPositionX = (double *)calloc(ssize, sizeof(double));

		j = (int)(5.0 * sigma + 0.5);
		if (j >= 1024 / 2)
		{
			// Error("SearchHighRes", "Too large sigma");
			return -1;
		}

		if (markov == true)
		{
			if (averWindow <= 0)
			{
				// Error("SearchHighRes", "Averanging window must be positive");
				return -2;
			}
		}

		if (backgroundRemove == true)
		{
			if (ssize < 2 * numberIterations + 1)
			{
				// Error("SearchHighRes", "Too large clipping window");
				return -3;
			}
		}

		k = int(2 * sigma + 0.5);
		if (k >= 2)
		{
			for (i = 0; i < k; i++)
			{
				a = i, b = source[i];
				m0low += 1, m1low += a, m2low += a * a, l0low += b, l1low += a * b;
			}
			detlow = m0low * m2low - m1low * m1low;
			if (detlow != 0)
				l1low = (-l0low * m1low + l1low * m0low) / detlow;

			else
				l1low = 0;
			if (l1low > 0)
				l1low = 0;
		}

		else
		{
			l1low = 0;
		}

		i = (int)(7 * sigma + 0.5);
		i = 2 * i;
		double *working_space = new double[7 * (ssize + i)];
		for (j = 0; j < 7 * (ssize + i); j++)
			working_space[j] = 0;
		for (i = 0; i < size_ext; i++)
		{
			if (i < shift)
			{
				a = i - shift;
				working_space[i + size_ext] = source[0] + l1low * a;
				if (working_space[i + size_ext] < 0)
					working_space[i + size_ext] = 0;
			}

			else if (i >= ssize + shift)
			{
				a = i - (ssize - 1 + shift);
				working_space[i + size_ext] = source[ssize - 1];
				if (working_space[i + size_ext] < 0)
					working_space[i + size_ext] = 0;
			}

			else
				working_space[i + size_ext] = source[i - shift];
		}

		if (backgroundRemove == true)
		{
			for (i = 1; i <= numberIterations; i++)
			{
				for (j = i; j < size_ext - i; j++)
				{
					if (markov == false)
					{
						a = working_space[size_ext + j];
						b = (working_space[size_ext + j - i] + working_space[size_ext + j + i]) / 2.0;
						if (b < a)
							a = b;

						working_space[j] = a;
					}

					else
					{
						a = working_space[size_ext + j];
						av = 0;
						men = 0;
						for (w = j - bw; w <= j + bw; w++)
						{
							if (w >= 0 && w < size_ext)
							{
								av += working_space[size_ext + w];
								men += 1;
							}
						}
						av = av / men;
						b = 0;
						men = 0;
						for (w = j - i - bw; w <= j - i + bw; w++)
						{
							if (w >= 0 && w < size_ext)
							{
								b += working_space[size_ext + w];
								men += 1;
							}
						}
						b = b / men;
						c = 0;
						men = 0;
						for (w = j + i - bw; w <= j + i + bw; w++)
						{
							if (w >= 0 && w < size_ext)
							{
								c += working_space[size_ext + w];
								men += 1;
							}
						}
						c = c / men;
						b = (b + c) / 2;
						if (b < a)
							av = b;
						working_space[j] = av;
					}
				}
				for (j = i; j < size_ext - i; j++)
					working_space[size_ext + j] = working_space[j];
			}
			for (j = 0; j < size_ext; j++)
			{
				if (j < shift)
				{
					a = j - shift;
					b = source[0] + l1low * a;
					if (b < 0)
						b = 0;
					working_space[size_ext + j] = b - working_space[size_ext + j];
				}

				else if (j >= ssize + shift)
				{
					a = j - (ssize - 1 + shift);
					b = source[ssize - 1];
					if (b < 0)
						b = 0;
					working_space[size_ext + j] = b - working_space[size_ext + j];
				}

				else
				{
					working_space[size_ext + j] = source[j - shift] - working_space[size_ext + j];
				}
			}
			for (j = 0; j < size_ext; j++)
			{
				if (working_space[size_ext + j] < 0)
					working_space[size_ext + j] = 0;
			}
		}

		for (i = 0; i < size_ext; i++)
		{
			working_space[i + 6 * size_ext] = working_space[i + size_ext];
		}

		if (markov == true)
		{
			for (j = 0; j < size_ext; j++)
				working_space[2 * size_ext + j] = working_space[size_ext + j];
			xmin = 0, xmax = size_ext - 1;
			for (i = 0, maxch = 0; i < size_ext; i++)
			{
				working_space[i] = 0;
				if (maxch < working_space[2 * size_ext + i])
					maxch = working_space[2 * size_ext + i];
				plocha += working_space[2 * size_ext + i];
			}
			if (maxch == 0)
			{
				delete[] working_space;
				return 0;
			}

			nom = 1;
			working_space[xmin] = 1;
			for (i = xmin; i < xmax; i++)
			{
				nip = working_space[2 * size_ext + i] / maxch;
				nim = working_space[2 * size_ext + i + 1] / maxch;
				sp = 0, sm = 0;
				for (l = 1; l <= averWindow; l++)
				{
					if ((i + l) > xmax)
						a = working_space[2 * size_ext + xmax] / maxch;

					else
						a = working_space[2 * size_ext + i + l] / maxch;

					b = a - nip;
					if (a + nip <= 0)
						a = 1;

					else
						a = sqrt(a + nip);

					b = b / a;
					b = exp(b);
					sp = sp + b;
					if ((i - l + 1) < xmin)
						a = working_space[2 * size_ext + xmin] / maxch;

					else
						a = working_space[2 * size_ext + i - l + 1] / maxch;

					b = a - nim;
					if (a + nim <= 0)
						a = 1;

					else
						a = sqrt(a + nim);

					b = b / a;
					b = exp(b);
					sm = sm + b;
				}
				a = sp / sm;
				a = working_space[i + 1] = working_space[i] * a;
				nom = nom + a;
			}
			for (i = xmin; i <= xmax; i++)
			{
				working_space[i] = working_space[i] / nom;
			}
			for (j = 0; j < size_ext; j++)
				working_space[size_ext + j] = working_space[j] * plocha;
			for (j = 0; j < size_ext; j++)
			{
				working_space[2 * size_ext + j] = working_space[size_ext + j];
			}
			if (backgroundRemove == true)
			{
				for (i = 1; i <= numberIterations; i++)
				{
					for (j = i; j < size_ext - i; j++)
					{
						a = working_space[size_ext + j];
						b = (working_space[size_ext + j - i] + working_space[size_ext + j + i]) / 2.0;
						if (b < a)
							a = b;
						working_space[j] = a;
					}
					for (j = i; j < size_ext - i; j++)
						working_space[size_ext + j] = working_space[j];
				}
				for (j = 0; j < size_ext; j++)
				{
					working_space[size_ext + j] = working_space[2 * size_ext + j] - working_space[size_ext + j];
				}
			}
		}
		// deconvolution starts
		area = 0;
		lh_gold = -1;
		posit = 0;
		maximum = 0;
		// generate response vector
		for (i = 0; i < size_ext; i++)
		{
			lda = (double)i - 3 * sigma;
			lda = lda * lda / (2 * sigma * sigma);
			j = (int)(1000 * exp(-lda));
			lda = j;
			if (lda != 0)
				lh_gold = i + 1;

			working_space[i] = lda;
			area = area + lda;
			if (lda > maximum)
			{
				maximum = lda;
				posit = i;
			}
		}
		// read source vector
		for (i = 0; i < size_ext; i++)
			working_space[2 * size_ext + i] = fabs(working_space[size_ext + i]);
		// create matrix at*a(vector b)
		i = lh_gold - 1;
		if (i > size_ext)
			i = size_ext;

		imin = -i, imax = i;
		for (i = imin; i <= imax; i++)
		{
			lda = 0;
			jmin = 0;
			if (i < 0)
				jmin = -i;
			jmax = lh_gold - 1 - i;
			if (jmax > (lh_gold - 1))
				jmax = lh_gold - 1;

			for (j = jmin; j <= jmax; j++)
			{
				ldb = working_space[j];
				ldc = working_space[i + j];
				lda = lda + ldb * ldc;
			}
			working_space[size_ext + i - imin] = lda;
		}
		// create vector p
		i = lh_gold - 1;
		imin = -i, imax = size_ext + i - 1;
		for (i = imin; i <= imax; i++)
		{
			lda = 0;
			for (j = 0; j <= (lh_gold - 1); j++)
			{
				ldb = working_space[j];
				k = i + j;
				if (k >= 0 && k < size_ext)
				{
					ldc = working_space[2 * size_ext + k];
					lda = lda + ldb * ldc;
				}
			}
			working_space[4 * size_ext + i - imin] = lda;
		}
		// move vector p
		for (i = imin; i <= imax; i++)
			working_space[2 * size_ext + i - imin] = working_space[4 * size_ext + i - imin];
		// initialization of resulting vector
		for (i = 0; i < size_ext; i++)
			working_space[i] = 1;
		// START OF ITERATIONS
		for (lindex = 0; lindex < deconIterations; lindex++)
		{
			for (i = 0; i < size_ext; i++)
			{
				if (fabs(working_space[2 * size_ext + i]) > 0.00001 && fabs(working_space[i]) > 0.00001)
				{
					lda = 0;
					jmin = lh_gold - 1;
					if (jmin > i)
						jmin = i;

					jmin = -jmin;
					jmax = lh_gold - 1;
					if (jmax > (size_ext - 1 - i))
						jmax = size_ext - 1 - i;

					for (j = jmin; j <= jmax; j++)
					{
						ldb = working_space[j + lh_gold - 1 + size_ext];
						ldc = working_space[i + j];
						lda = lda + ldb * ldc;
					}
					ldb = working_space[2 * size_ext + i];
					if (lda != 0)
						lda = ldb / lda;

					else
						lda = 0;

					ldb = working_space[i];
					lda = lda * ldb;
					working_space[3 * size_ext + i] = lda;
				}
			}
			for (i = 0; i < size_ext; i++)
			{
				working_space[i] = working_space[3 * size_ext + i];
			}
		}
		// shift resulting spectrum
		for (i = 0; i < size_ext; i++)
		{
			lda = working_space[i];
			j = i + posit;
			j = j % size_ext;
			working_space[size_ext + j] = lda;
		}
		// write back resulting spectrum
		maximum = 0, maximum_decon = 0;
		j = lh_gold - 1;
		for (i = 0; i < size_ext - j; i++)
		{
			if (i >= shift && i < ssize + shift)
			{
				working_space[i] = area * working_space[size_ext + i + j];
				if (maximum_decon < working_space[i])
					maximum_decon = working_space[i];
				if (maximum < working_space[6 * size_ext + i])
					maximum = working_space[6 * size_ext + i];
			}

			else
				working_space[i] = 0;
		}
		lda = 1;
		if (lda > threshold)
			lda = threshold;
		lda = lda / 100;

		// searching for peaks in deconvolved spectrum
		for (i = 1; i < size_ext - 1; i++)
		{
			if (working_space[i] > working_space[i - 1] && working_space[i] > working_space[i + 1])
			{
				if (i >= shift && i < ssize + shift)
				{
					if (working_space[i] > lda * maximum_decon && working_space[6 * size_ext + i] > threshold * maximum / 100.0)
					{
						for (j = i - 1, a = 0, b = 0; j <= i + 1; j++)
						{
							a += (double)(j - shift) * working_space[j];
							b += working_space[j];
						}
						a = a / b;
						if (a < 0)
							a = 0;

						if (a >= ssize)
							a = ssize - 1;
						if (peak_index == 0)
						{
							fPositionX[0] = a;
							peak_index = 1;
						}

						else
						{
							for (j = 0, priz = 0; j < peak_index && priz == 0; j++)
							{
								if (working_space[6 * size_ext + shift + (int)a] > working_space[6 * size_ext + shift + (int)fPositionX[j]])
									priz = 1;
							}
							if (priz == 0)
							{
								if (j < fMaxPeaks)
								{
									fPositionX[j] = a;
								}
							}

							else
							{
								for (k = peak_index; k >= j; k--)
								{
									if (k < fMaxPeaks)
									{
										fPositionX[k] = fPositionX[k - 1];
									}
								}
								fPositionX[j - 1] = a;
							}
							if (peak_index < fMaxPeaks)
								peak_index += 1;
						}
					}
				}
			}
		}

		for (i = 0; i < ssize; i++)
			destVector[i] = working_space[i + shift];
		delete[] working_space;
		// int fNPeaks = peak_index;
		// if (peak_index == fMaxPeaks)
		// 	Warning("SearchHighRes", "Peak buffer full");
		return peak_index;
	}

	static PyObject *Ransac(PyObject *self, PyObject *args)
	{
		// PyObject* list;
		PyArrayObject *p;
		int number_it, min_inliers, mode;
		double min_dist;
		NpyIter *in_iter;
		// if(!PyArg_ParseTuple(args, "Oidii", &list, &number_it, &min_dist, &min_inliers, &mode)){
		if (!PyArg_ParseTuple(args, "O!idii", &PyArray_Type, &p, &number_it, &min_dist, &min_inliers, &mode))
		{
			return NULL;
		}
		if (PyArray_DESCR(p)->type_num != NPY_DOUBLE)
		{
			// std::cout << "Type np.float64 expected for p array.";
			PyErr_SetString(PyExc_TypeError, "Type np.float64 expected for array.");
			return NULL;
		}
		// else std::cout << "Type np.float64 expected for p array is correct.";

		if (PyArray_NDIM(p) != 2)
		{
			// std::cout << "p must be a 2 dimensionnal array.";
			PyErr_SetString(PyExc_TypeError, "Array must be 2 dimensional.");
			return NULL;
		}
		// Py_ssize_t size = PyList_GET_SIZE(list);
		int rows = PyArray_DIM(p, 0);
		int cols = PyArray_DIM(p, 1);
		std::vector<double> charge(rows);
		std::vector<std::vector<double>> data;
		std::vector<long int> inliers;
		in_iter = NpyIter_New(p, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
		double **in_dataptr = (double **)NpyIter_GetDataPtrArray(in_iter);
		NpyIter_IterNextFunc *in_iternext = NpyIter_GetIterNext(in_iter, NULL);
		// for(int i = 0; i < (int) size; i++){
		for (int i = 0; i < rows; i++)
		{
			// PyObject* Point3D = PyList_GetItem(list, i);
			std::vector<double> PartVec(4);
			for (int j = 0; j < cols; j++)
			{
				// PartVec[j] = PyFloat_AsDouble(PyList_GetItem(Point3D, j));
				PartVec[j] = **in_dataptr;
				in_iternext(in_iter);
			}
			// charge[i] = PyFloat_AsDouble(PyList_GetItem(Point3D, 3));
			charge[i] = PartVec[4];
			data.push_back(PartVec);
		};
		NpyIter_Deallocate(in_iter);
		PyObject *PyInliers = PyList_New(0);
		PyObject *PyVersor = PyList_New(0);
		PyObject *PyPb = PyList_New(0);
		pRansac(data, PyInliers, PyVersor, PyPb, charge, number_it, min_dist, mode, min_inliers);
		PyObject *NPInliers;
		if (PyList_Size(PyInliers) == 1)
			NPInliers = PyArray_FROM_OTF(PyInliers, NPY_INT, NPY_ARRAY_IN_ARRAY);
		else
			NPInliers = PyArray_FROM_OTF(PyInliers, NPY_OBJECT, NPY_ARRAY_IN_ARRAY);
		Py_DecRef(PyInliers);
		return Py_BuildValue("(OOO)", NPInliers,
							 PyArray_FROM_OTF(PyVersor, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
							 PyArray_FROM_OTF(PyPb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
	}

	static PyObject *Fit3D(PyObject *self, PyObject *args)
	{
		// PyObject* list;
		PyArrayObject *p;
		PyObject *PyVersor = PyList_New(3);
		PyObject *PyPoint = PyList_New(3);
		NpyIter *in_iter;
		if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &p))
			return NULL;
		// if(!PyArg_ParseTuple(args, "O", &list)) return NULL;
		// long int size = PyList_Size(list);
		if (PyArray_DESCR(p)->type_num != NPY_DOUBLE)
		{
			// std::cout << "Type np.float64 expected for p array.";
			PyErr_SetString(PyExc_TypeError, "Type np.float64 expected for array.");
			return NULL;
		}
		// else std::cout << "Type np.float64 expected for p array is correct.";

		if (PyArray_NDIM(p) != 2)
		{
			// std::cout << "p must be a 2 dimensionnal array.";
			PyErr_SetString(PyExc_TypeError, "Array must be 2 dimensional.");
			return NULL;
		}
		int rows = PyArray_DIM(p, 0);
		int cols = PyArray_DIM(p, 1);
		std::vector<double> vX(rows);
		std::vector<double> vY(rows);
		std::vector<double> vZ(rows);
		std::vector<double> vQ(rows);
		std::vector<double> versor(3);
		std::vector<double> pb(3);
		in_iter = NpyIter_New(p, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
		double **in_dataptr = (double **)NpyIter_GetDataPtrArray(in_iter);
		NpyIter_IterNextFunc *in_iternext = NpyIter_GetIterNext(in_iter, NULL);
		// for(int i = 0; i < size; i++){
		for (int i = 0; i < rows; i++)
		{
			// PyObject* Point3D = PyList_GetItem(list, i);
			std::vector<double> PartVec(4);
			for (int j = 0; j < 4; j++)
			{
				// PartVec[j] = PyFloat_AsDouble(PyList_GetItem(Point3D, j));
				PartVec[j] = **in_dataptr;
				in_iternext(in_iter);
			}
			vX[i] = PartVec[0];
			vY[i] = PartVec[1];
			vZ[i] = PartVec[2];
			vQ[i] = PartVec[3];
		}
		Fit3D2(vX, vY, vZ, vQ, versor, pb);
		for (int i = 0; i < 3; i++)
		{
			PyList_SetItem(PyVersor, i, PyFloat_FromDouble(versor[i]));
			PyList_SetItem(PyPoint, i, PyFloat_FromDouble(pb[i]));
		}
		NpyIter_Deallocate(in_iter);
		// return Py_BuildValue("(OO)", PyVersor, PyPoint);
		return Py_BuildValue("(OO)", PyArray_FROM_OTF(PyVersor, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
							 PyArray_FROM_OTF(PyPoint, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
	}

	static PyObject *Background(PyObject *self, PyObject *args)
	{
		PyArrayObject *p;
		int number_it, direction, filter_order, smooth_window;
		bool smoothing, compton;
		NpyIter *in_iter;
		if (!PyArg_ParseTuple(args, "O!iiibib", &PyArray_Type, &p, &number_it, &direction, &filter_order, &smoothing, &smooth_window, &compton))
			return NULL;
		if (PyArray_DESCR(p)->type_num != NPY_DOUBLE)
		{
			PyErr_SetString(PyExc_TypeError, "Type np.float64 expected for array.");
			return NULL;
		}
		if (PyArray_NDIM(p) > 1)
		{
			PyErr_SetString(PyExc_TypeError, "Array must be 1 dimensional.");
			return NULL;
		}
		int size = PyArray_DIM(p, 0);
		if (size < 1)
		{
			PyErr_SetString(PyExc_TypeError, "Array must have at least one element.");
			return NULL;
		}
		in_iter = NpyIter_New(p, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
		double **in_dataptr = (double **)NpyIter_GetDataPtrArray(in_iter);
		NpyIter_IterNextFunc *in_iternext = NpyIter_GetIterNext(in_iter, NULL);
		double *spectrum = new double[size];
		for (int i = 0; i < size; i++)
		{
			spectrum[i] = **in_dataptr;
			in_iternext(in_iter);
		}
		NpyIter_Deallocate(in_iter);
		int res = Background_root(spectrum, size, number_it, direction, filter_order, smoothing, smooth_window, compton);
		if (res < 0)
		{
			if (res == -1)
			{
				PyErr_SetString(PyExc_TypeError, "Too Large Clipping Window.");
				return NULL;
			}
			PyErr_SetString(PyExc_TypeError, "Incorrect width of smoothing window.");
			return NULL;
		}
		PyObject *PySpectrum = PyList_New(size);
		for (int i = 0; i < size; i++)
		{
			PyList_SetItem(PySpectrum, i, PyFloat_FromDouble(spectrum[i]));
		}
		delete[] spectrum;
		PyObject *PySpectrum_np = PyArray_FROM_OTF(PySpectrum, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
		Py_DECREF(PySpectrum);
		return Py_BuildValue("O", PySpectrum_np);
	}

	static PyObject *SearchHighRes(PyObject *self, PyObject *args)
	{
		PyArrayObject *p;
		int number_it, aver_window;
		double sigma, threshold;
		bool bkg_remove, markov;
		NpyIter *in_iter;
		if (!PyArg_ParseTuple(args, "O!ddbibi", &PyArray_Type, &p, &sigma, &threshold, &bkg_remove, &number_it, &markov, &aver_window))
			return NULL;
		if (PyArray_DESCR(p)->type_num != NPY_DOUBLE)
		{
			PyErr_SetString(PyExc_TypeError, "Type np.float64 expected for array.");
			return NULL;
		}
		if (PyArray_NDIM(p) > 1)
		{
			PyErr_SetString(PyExc_TypeError, "Array must be 1 dimensional.");
			return NULL;
		}
		int size = PyArray_DIM(p, 0);
		if (size < 1)
		{
			PyErr_SetString(PyExc_TypeError, "Array must have at least one element.");
			return NULL;
		}
		if (sigma < 0)
		{
			PyErr_SetString(PyExc_TypeError, "Invalid sigma, must be greater than or equal to 1.");
			return NULL;
		}
		if (threshold <= 0 || threshold >= 100)
		{
			PyErr_SetString(PyExc_TypeError, "Invalid threshold, must be greater than or equal to 0.");
			return NULL;
		}
		in_iter = NpyIter_New(p, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
		double **in_dataptr = (double **)NpyIter_GetDataPtrArray(in_iter);
		NpyIter_IterNextFunc *in_iternext = NpyIter_GetIterNext(in_iter, NULL);
		double *source = new double[size];
		double *response = new double[size];
		double *pos_peaks = new double[size];
		for (int i = 0; i < size; i++)
		{
			source[i] = **in_dataptr;
			in_iternext(in_iter);
		}
		NpyIter_Deallocate(in_iter);
		int res = SearchHighRes_root(source, response, pos_peaks, size, sigma, threshold, bkg_remove, number_it, markov, aver_window);
		if (res < 0)
		{
			if (res == -1)
			{
				PyErr_SetString(PyExc_TypeError, "Too large sigma.");
				return NULL;
			}
			else if (res == -2)
			{
				PyErr_SetString(PyExc_TypeError, "Averanging window must be positive.");
				return NULL;
			}
			PyErr_SetString(PyExc_TypeError, "Too large clipping window.");
			return NULL;
		}
		PyObject *PyResponse = PyList_New(size);
		PyObject *PyPosPeaks = PyList_New(res);
		for (int i = 0; i < size; i++)
		{
			PyList_SetItem(PyResponse, i, PyFloat_FromDouble(response[i]));
		}
		for (int i = 0; i < res; i++)
		{
			PyList_SetItem(PyPosPeaks, i, PyFloat_FromDouble(pos_peaks[i]));
		}
		delete[] source;
		delete[] response;
		delete[] pos_peaks;
		PyObject *PyResponse_np = PyArray_FROM_OTF(PyResponse, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
		Py_DECREF(PyResponse);
		PyObject *PyPosPeaks_np = PyArray_FROM_OTF(PyPosPeaks, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
		Py_DECREF(PyPosPeaks);
		return Py_BuildValue("(OO)", PyResponse_np, PyPosPeaks_np);
	}

	static PyObject *version(PyObject *self)
	{
		return Py_BuildValue("s", "0.1");
	}

	static PyObject *get_peaks(PyObject *self, PyObject *args)
	{
		PyArrayObject *p;
		NpyIter *in_iter;
		int i, j, k, l;
		std::vector<std::vector<double>> vec_peaks;
		if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &p))
		{
			std::cout << "Erro\n";
			return NULL;
		}

		if (PyArray_DESCR(p)->type_num != NPY_DOUBLE)
		{
			// std::cout << "Type np.float64 expected for p array.";
			PyErr_SetString(PyExc_TypeError, "Array de entrada deve ser do tipo float.");
			return NULL;
		}
		// else std::cout << "Type np.float64 expected for p array is correct.";

		if (PyArray_NDIM(p) != 2)
		{
			// std::cout << "p must be a 2 dimensionnal array.";
			PyErr_SetString(PyExc_TypeError, "Array de entrada deve ser bidimensional.");
			return NULL;
		}

		int size = PyArray_DIM(p, 0);
		int cols = PyArray_DIM(p, 1);
		in_iter = NpyIter_New(p, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
		double **in_dataptr = (double **)NpyIter_GetDataPtrArray(in_iter);
		NpyIter_IterNextFunc *in_iternext = NpyIter_GetIterNext(in_iter, NULL);

		std::vector<double> vec_array(cols);
		vec_peaks.reserve(size);
		for (i = 0; i < size; i++)
		{
			for (j = 0; j < cols; j++)
			{
				vec_array[j] = **in_dataptr;
				in_iternext(in_iter);
			}
			std::vector<double> vec_peaks_parcial;
			for (j = 0; j < 512; j++)
			{
				if (vec_array[j] > 0.5)
				{
					std::vector<int> indice_parcial;
					while (true)
					{
						if (vec_array[j] < 0.5)
						{
							break;
						}
						indice_parcial.push_back(j);
						j += 1;
					}
					if (indice_parcial.size() > 0)
					{
						double peso = 0., total = 0.;
						for (k = 0; k < indice_parcial.size(); k++)
						{
							// std::cout << indice_parcial[k] + 512 << "\n";
							peso += vec_array[indice_parcial[k] + 512] * indice_parcial[k];
							total += vec_array[indice_parcial[k] + 512];
						}
						vec_peaks_parcial.push_back((double)peso / total);
					}
					indice_parcial.clear();
				}
			}
			vec_peaks.push_back(vec_peaks_parcial);
			vec_peaks_parcial.clear();
		}
		PyObject *peaks = PyList_New(0);
		for (i = 0; i < size; i++)
		{
			PyObject *peaks_parcial = PyList_New(0);
			for (j = 0; j < vec_peaks[i].size(); j++)
				PyList_Append(peaks_parcial, PyFloat_FromDouble(vec_peaks[i][j]));
			PyList_Append(peaks, PyArray_FROM_OTF(peaks_parcial, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
		}
		NpyIter_Deallocate(in_iter);
		return Py_BuildValue("O", PyArray_FROM_OTF(peaks, NPY_OBJECT, NPY_ARRAY_IN_ARRAY));
	}

	static PyObject *teste_numpy(PyObject *self, PyObject *args)
	{
		PyArrayObject *p;
		PyObject *a;
		NpyIter *in_iter;
		if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &p))
		{
			std::cout << "Erro\n";
			return NULL;
		}
		if (PyArray_DESCR(p)->type_num != NPY_DOUBLE)
		{
			// std::cout << "Type np.float64 expected for p array.";
			PyErr_SetString(PyExc_TypeError, "Type np.float64 expected for p array.");
			return NULL;
		}
		// else std::cout << "Type np.float64 expected for p array is correct.";

		if (PyArray_NDIM(p) != 2)
		{
			// std::cout << "p must be a 2 dimensionnal array.";
			PyErr_SetString(PyExc_TypeError, "p must be a 2 dimensionnal array.");
			return NULL;
		}

		a = PyArray_NewLikeArray(p, NPY_ANYORDER, NULL, 0);
		if (a == NULL)
		{
			std::cout << "Deu ruim pra criar o array de retorno";
			return NULL;
		}
		PyArray_FILLWBYTE(a, 0);

		int rows = PyArray_DIM(p, 0);
		int cols = PyArray_DIM(p, 1);
		std::cout << "\nNumero de linhas = " << rows << ". Numero de colunas  = " << cols << ".\n";
		in_iter = NpyIter_New(p, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
		double **in_dataptr = (double **)NpyIter_GetDataPtrArray(in_iter);
		NpyIter_IterNextFunc *in_iternext = NpyIter_GetIterNext(in_iter, NULL);
		int i = 0;
		for (int linha = 0; linha < rows; linha++)
		{
			for (int coluna = 0; coluna < cols; coluna++)
			{
				std::cout << "Linha " << linha << ". Coluna " << coluna << " = " << **in_dataptr << "\n";
				in_iternext(in_iter);
			}
		}
		NpyIter_Deallocate(in_iter);
		// do {
		//     std::cout<<"Iteracao "<< i << ". Valor = " << **in_dataptr << "\n";
		// 	i += 1;
		// } while(in_iternext(in_iter));

		return Py_BuildValue("s", "Ok.");
	}

	static PyMethodDef myMethods[] = {
		// {"version", (PyCFunction)version, METH_NOARGS, "Returns the version."},
		{"ransac", Ransac, METH_VARARGS, "Calculates all 3D lines."},
		{"fit3D", Fit3D, METH_VARARGS, "Calculates the best 3D line."},
		{"get_peaks", get_peaks, METH_VARARGS, "Retorna picos da saída da rede neural."},
		{"background", Background, METH_VARARGS, "Calculates the background."},
		{"search_high_res", SearchHighRes, METH_VARARGS, "One-dimensional high-resolution peak search function."},
		{NULL, NULL, 0, NULL}};

	static struct PyModuleDef tpc_utils_ = {
		PyModuleDef_HEAD_INIT,
		"tpc_utils_",
		"Useful methods for TPC data.",
		-1,
		myMethods};

	PyMODINIT_FUNC PyInit_tpc_utils_(void)
	{
		import_array();
		return PyModule_Create(&tpc_utils_);
	}

	// END
}