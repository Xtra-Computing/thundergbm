/*
 * Communicator.h
 *
 *	@brief: a class for receiving and sending data pack
 *  Created on: 23/09/2014
 *      Author: Zeyi Wen
 */

#ifndef COMMUNICATOR_H_
#define COMMUNICATOR_H_

class DataPack
{
public:
	int nNumofSeg;
	int *pnSizeofSeg;
	char **ypData;
};


class Communicator
{
protected:
	static const int DATA_BUFFER = 102400 + 1;		//for storing data from clients and workers
	static const int ENVELOP_DEL_BUFFER = 64 + 1;
	static const int ID_BUFFER = 1024 + 1;			//for storing IDs (e.g., worker id, client id, task id)

public:
	Communicator(){}
	virtual ~Communicator(){}

	virtual void SendEmpty(void *comm_socket){};
	virtual void RecvEmpty(void *comm_socket){};

	void RecvDataPack(void*, DataPack *&taskData);
	void SendMore(void *comm_socket, DataPack *dataPack);
	void SendLast(void *comm_socket, DataPack *dataPack);
	void SendDataPack(void *comm_socket, DataPack *dataPack, bool bSendMore);

private:
	void RecvDataHeader(void *taskReceiver, int &nNumofSegments, int *&pnSizeofSemgnet);
	void RecvNumofSeg(void *taskReceiver, int &nNumofSegments);
	void RecvSizeofSeg(void *taskReceiver, int nNumofSegments, int *&pnSizeofSemgnet);
	void RecvData(void *taskReceiver, char *&pRecvedData, int);
};

#endif /* COMMUNICATOR_H_ */
