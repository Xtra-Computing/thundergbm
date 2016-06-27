/*
 * GDPair.h
 *
 *  Created on: 12 Apr 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef GDPAIR_H_
#define GDPAIR_H_

struct gdpair
{
  /*! \brief gradient statistics */
  double grad;
  /*! \brief second order gradient statistics */
  double hess;
  gdpair(void) {grad = 0; hess = 0;}
  gdpair(double grad, double hess) : grad(grad), hess(hess) {}
};



#endif /* GDPAIR_H_ */
