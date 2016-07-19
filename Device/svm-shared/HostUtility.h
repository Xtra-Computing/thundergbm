/*
 * HostUtility.h
 *
 *  Created on: 19 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef HOSTUTILITY_H_
#define HOSTUTILITY_H_


#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#define Ceil(a, b) (a%b!=0)?((a/b)+1):(a/b)


#endif /* HOSTUTILITY_H_ */
