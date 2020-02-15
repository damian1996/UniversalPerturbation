array=( Centipede Phoenix ChopperCommand Gopher Krull YarsRevenge Seaquest Breakout Atlantis
        Pong Assault Solaris UpNDown DoubleDunk Tennis StarGunner Zaxxon Qbert Gravitar )


for i in "${array[@]}"
do
    python3 make_transfer_table_with_other_pert.py --env $i --algo $1
done
