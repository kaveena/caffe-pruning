#!/bin/bash

if [[ "$PATH" != *`echo $(realpath install/bin)`* ]]
then
  export PATH+=:$(realpath install/bin)
fi

if [[ "$PYTHONPATH" != *`echo $(realpath install/python)`* ]]
then
  export PYTHONPATH+=:$(realpath install/python)
fi
