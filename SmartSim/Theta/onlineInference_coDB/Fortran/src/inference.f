      module ssim

      use smartredis_client, only : client_type      

      type(client_type) :: client
      integer nnDB

      end module ssim

c ==================================================

      subroutine init_client(myrank)

      use iso_c_binding
      use ssim
      implicit none
      include "mpif.h"

      integer myrank, iSSres, SRNoError

      SRNoError = 0

c     Initialize SmartRedis clients 
      if (myrank.eq.0) write(*,*) 'Initializing SmartRedis clients ... '
      if (nnDB.eq.1) then
         iSSres = client%initialize(.false.) ! NOT using a clustered database (DB on 1 node only)
      else
         iSSres = client%initialize(.true.) ! using a clustered database (DB on multiple nodes)
      endif
      if (iSSres.ne.SRNoError) write(*,*) 
     &             "ERROR: client%initialize failed on rank ",myrank



      end subroutine init_client

c ==================================================

      program  inference

      use iso_c_binding
      use fortran_c_interop
      use ssim
      implicit none
      include "mpif.h"

      real*8, allocatable, dimension (:,:) :: inf_data
      real*8, allocatable, dimension (:,:) :: pred_data
      real*8, allocatable, dimension (:,:) :: truth_data
      real*8 xmin, xmax, x
      integer nSamples, seed
      integer its, numts, i
      integer myrank, comm_size, ierr, tag, status(MPI_STATUS_SIZE), 
     &        nproc
      integer iSSres, SRNoError
      character*255 inf_key, pred_key
      logical im_exst
      character(len=255), dimension(1) :: inputs
      character(len=255), dimension(1) :: outputs

      call MPI_INIT(ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, comm_size, ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
      nproc = comm_size
      if (myrank.eq.0) write(*,*) "Initialized ",nproc," MPI procs"

c     Initialize SmartRedis clients 
      SRNoError = 0
      nnDB = 1 
      call init_client(myrank)
      call MPI_Barrier(MPI_COMM_WORLD,ierr)
      if (myrank.eq.0) write(*,*) 'SmartSim clients initialized'


c     Load the model on the database
      if (myrank.eq.0.or.myrank.eq.32) then
         iSSres = client%set_model_from_file("model", "model_jit.pt",
     &                                   "TORCH", "CPU")
         if (iSSres.ne.SRNoError) then 
           write(*,*) 
     &       "ERROR: client%set_model_from_file failed on rank ", myrank
         else
            write(*,*) "Uploaded model to database from rank ", myrank
         endif
      endif
      call MPI_Barrier(MPI_COMM_WORLD,ierr)


c     Set parameters for array of random numbers to be set as inference data
c     In this example we create inference data for a simple function
c     y=f(x), which has 1 input (x) and 1 output (y)
c     The domain for the function is from 0 to 10
c     Tha inference data is obtained from a uniform distribution over the domain
      nSamples = 64
      allocate(inf_data(nSamples,1))
      allocate(pred_data(nSamples,1))
      allocate(truth_data(nSamples,1))
      seed = myrank+1
      call RANDOM_SEED(seed)
      if (myrank.lt.32) then
         xmin = 0.0
         xmax = 5.0
      else if (myrank.lt.64) then
         xmin = 5.0
         xmax = 10.0
      endif


c     Generate the key for the inference data
c     The key will be tagged with the rank ID
      inf_key = "y."
      pred_key = "p."
      if (myrank.lt.10) then
         write (inf_key, "(A2,I1)") trim(inf_key), myrank
         write (pred_key, "(A2,I1)") trim(pred_key), myrank
      elseif (myrank.lt.100) then
         write (inf_key, "(A2,I2)") trim(inf_key), myrank
         write (pred_key, "(A2,I2)") trim(pred_key), myrank
      elseif (myrank.lt.1000) then
         write (inf_key, "(A2,I3)") trim(inf_key), myrank
         write (pred_key, "(A2,I3)") trim(pred_key), myrank
      endif


c     Open file to write predictions
      if (myrank.eq.0) then
         open(unit=10,file = "predictions_node1.dat")
      endif
      if (myrank.eq.32) then
         open(unit=11,file = "predictions_node2.dat")
      endif


c     Emulate integration of PDEs with a do loop
      numts = 2
      do its=1,numts
         ! sleep for a few seconds to emulate the time required by PDE integration
         call sleep (10)

         ! generate the inference data for the polynomial y=f(x)=x**2 + 3*x + 1
         ! place output in first column, input in second column
         do i=1,nSamples
            call RANDOM_NUMBER(x)
            x = xmin + (xmax-xmin)*x
            inf_data(i,1) = x
            truth_data(i,1) = x**2 + 3*x +1
         enddo

         ! Send the inference data
         if (myrank.eq.0) write(*,*) 
     &            'Sending inference data to database with key ',
     &            trim(inf_key), ' and shape ',
     &            shape(inf_data)
         iSSres = client%put_tensor(trim(inf_key), inf_data, 
     &                          shape(inf_data))
         if (iSSres.ne.SRNoError) write(*,*)
     &       "ERROR: client%put_tensor failed on rank ",myrank
         call MPI_Barrier(MPI_COMM_WORLD,ierr)
         if (myrank.eq.0) write(*,*) 'Finished sending inference data'

         ! Evaluate the model on the database
         inputs(1) = inf_key
         outputs(1) = pred_key
         iSSres = client%run_model('model', inputs, outputs)
         if (iSSres.ne.SRNoError) write(*,*)
     &    "ERROR: client%run_model failed on rank ",myrank
         call MPI_Barrier(MPI_COMM_WORLD,ierr)
         if (myrank.eq.0) write(*,*) 'Finished evaluating model'

         ! Retreive the predictions
         iSSres = client%unpack_tensor(trim(pred_key), pred_data,
     &                             shape(pred_data))
         if (iSSres.ne.SRNoError) write(*,*)
     &       "ERROR: client%unpack_tensor failed on rank ",myrank
         call MPI_Barrier(MPI_COMM_WORLD,ierr)
         if (myrank.eq.0) write(*,*) 'Finished retreiving predictions'

         ! Write the inference, prediction and truth data to file for plotting
         if (myrank.eq.0.or.myrank.eq.32) then
            do i=1,nSamples
               if (myrank.eq.0) then
                  write(10,*) inf_data(i,1), pred_data(i,1), 
     &                     truth_data(i,1)
               else if (myrank.eq.32) then
                  write(11,*) inf_data(i,1), pred_data(i,1), 
     &                     truth_data(i,1)
               endif
            enddo
         endif

      enddo


c     Finilization stuff
      if (myrank.eq.0) write(*,*) "Exiting ... "
      if (myrank.eq.0) close(10)
      if (myrank.eq.32) close(11)
      deallocate(inf_data)
      deallocate(pred_data)
      deallocate(truth_data)
      call MPI_FINALIZE(ierr)

      end program inference
